"""
05_analysis_and_ml.py
Analysis and ML pipeline for the Stoke Health Atlas.

Analysis:
  - Antidepressant prescribing ratio Q1 vs Q5
  - Spearman IMD vs antidepressant rate
  - Spearman NO2 vs respiratory rate (synthetic AQ — illustrative)
  - Outlier GP practices (antidepressant z-score > 2.0)
  - Saves analysis_quintile and outlier_practices to DuckDB

ML:
  - XGBoost regressor: IMD19, imd_quintile, NO2, PM25 -> antidepressant items_per_1000
  - 5-fold CV if <50 rows, else 80/20 split
  - SHAP beeswarm + bar chart -> data/processed/
  - Model saved to data/processed/xgb_model.pkl
"""

import pickle
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
DB_PATH   = ROOT / "health_atlas.duckdb"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = db.execute("SELECT * FROM atlas_monthly").df()
    # Real-IMD practices only (not fallback Q3)
    df_real = df[df["imd19"].notna()].copy()
    print(f"atlas_monthly     : {len(df):,} rows, {df['practice_code'].nunique()} practices")
    print(f"Real-IMD subset   : {len(df_real):,} rows, {df_real['practice_code'].nunique()} practices")
    return df, df_real


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def analysis(df: pd.DataFrame, df_real: pd.DataFrame,
             db: duckdb.DuckDBPyConnection) -> dict:
    results = {}

    # ── A1. Antidepressant Q1 vs Q5 ratio ────────────────────────────────────
    ad = df_real[df_real["drug_category"] == "antidepressants"]
    q1_mean = ad[ad["imd_quintile"] == 1]["items_per_1000"].mean()
    q5_mean = ad[ad["imd_quintile"] == 5]["items_per_1000"].mean()
    ratio   = q1_mean / q5_mean if q5_mean else float("nan")
    results["q1_mean"]  = round(q1_mean, 1)
    results["q5_mean"]  = round(q5_mean, 1)
    results["ad_ratio"] = round(ratio, 2)
    print(f"\n[A1] Antidepressant prescribing")
    print(f"     Q1 mean items/1000 : {q1_mean:.1f}")
    print(f"     Q5 mean items/1000 : {q5_mean:.1f}")
    print(f"     Q1 / Q5 ratio      : {ratio:.2f}x")

    # ── A2. Spearman: IMD19 vs antidepressant rate ────────────────────────────
    # Aggregate to practice level (mean across months) to avoid repeated measures
    ad_prac = (
        ad.groupby("practice_code")
        .agg(imd19=("imd19", "first"), rate=("items_per_1000", "mean"))
        .dropna()
    )
    r_imd, p_imd = stats.spearmanr(ad_prac["imd19"], ad_prac["rate"])
    sig_imd = "significant" if p_imd < 0.05 else "not significant"
    results["r_imd"] = round(r_imd, 3)
    results["p_imd"] = round(p_imd, 4)
    print(f"\n[A2] Spearman: IMD19 vs antidepressant items/1000")
    print(f"     n={len(ad_prac)}  r={r_imd:.3f}  p={p_imd:.4f}  -> {sig_imd} at p<0.05")
    # IMD19 here is the national rank (1=most deprived, ~32844=least deprived).
    # r<0 means higher rank (less deprived) → lower prescribing — the expected direction.
    direction = ("Higher rank (less deprived) -> lower prescribing [expected]"
                 if r_imd < 0 else
                 "Higher rank (less deprived) -> higher prescribing [unexpected]")
    print(f"     {direction}")

    # ── A3. Spearman: NO2 vs respiratory rate ────────────────────────────────
    resp = df_real[df_real["drug_category"] == "respiratory"]
    resp_prac = (
        resp.groupby("practice_code")
        .agg(no2=("mean_no2", "mean"), rate=("items_per_1000", "mean"))
        .dropna()
    )
    r_no2, p_no2 = stats.spearmanr(resp_prac["no2"], resp_prac["rate"])
    sig_no2 = "significant" if p_no2 < 0.05 else "not significant"
    results["r_no2"] = round(r_no2, 3)
    results["p_no2"] = round(p_no2, 4)
    print(f"\n[A3] Spearman: NO2 vs respiratory items/1000  [SYNTHETIC AQ — illustrative]")
    print(f"     n={len(resp_prac)}  r={r_no2:.3f}  p={p_no2:.4f}  -> {sig_no2} at p<0.05")

    # ── A4. Outlier GP practices (antidepressant z-score > 2) ─────────────────
    ad_prac2 = (
        ad.groupby(["practice_code", "practice_name"])
        .agg(rate=("items_per_1000", "mean"), imd19=("imd19", "first"),
             imd_quintile=("imd_quintile", "first"))
        .reset_index()
        .dropna(subset=["rate"])
    )
    ad_prac2["z_score"] = (
        (ad_prac2["rate"] - ad_prac2["rate"].mean()) / ad_prac2["rate"].std()
    )
    outliers = ad_prac2[ad_prac2["z_score"] > 2.0].sort_values("z_score", ascending=False)
    results["n_outliers"] = len(outliers)
    print(f"\n[A4] Outlier practices (antidepressant z-score > 2.0)  [{len(outliers)} found]")
    for _, row in outliers.iterrows():
        print(f"     {row['practice_code']}  {row['practice_name'][:40]:<40}"
              f"  rate={row['rate']:.1f}  z={row['z_score']:.2f}  Q{int(row['imd_quintile'])}")

    # ── A5. Quintile × drug summary table → DuckDB ────────────────────────────
    quintile_summary = (
        df_real.groupby(["imd_quintile", "drug_category"])["items_per_1000"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"items_per_1000": "avg_rate"})
    )
    db.execute("CREATE OR REPLACE TABLE analysis_quintile AS SELECT * FROM quintile_summary")
    print(f"\n[A5] analysis_quintile saved: {len(quintile_summary)} rows")

    # Outlier table → DuckDB
    outlier_save = outliers[["practice_code", "practice_name", "rate",
                             "z_score", "imd19", "imd_quintile"]].copy()
    outlier_save.columns = ["practice_code", "practice_name",
                            "antidep_rate_per_1000", "z_score", "imd19", "imd_quintile"]
    db.execute("CREATE OR REPLACE TABLE outlier_practices AS SELECT * FROM outlier_save")
    print(f"[A5] outlier_practices saved: {len(outlier_save)} rows")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ML
# ═══════════════════════════════════════════════════════════════════════════════
def ml(df_real: pd.DataFrame) -> dict:
    results = {}

    # ── Build feature matrix ──────────────────────────────────────────────────
    # Keep monthly grain (practice × month) so NO2/PM25 vary across rows.
    # 128 real-IMD practices × 3 months = ~384 rows — large enough for CV.
    # Outlier specialist services (z>2 on practice mean rate) are excluded.
    ad_all = df_real[df_real["drug_category"] == "antidepressants"].copy()

    # Compute per-practice mean rate to identify outliers, then exclude those practices
    prac_mean = ad_all.groupby("practice_code")["items_per_1000"].mean()
    z_prac = (prac_mean - prac_mean.mean()) / prac_mean.std()
    outlier_codes = set(z_prac[z_prac > 2.0].index)
    feat = ad_all[~ad_all["practice_code"].isin(outlier_codes)].copy()
    feat = feat.dropna(subset=["items_per_1000", "imd19", "mean_no2", "mean_pm25"])
    print(f"     Outlier practices excluded : {len(outlier_codes)}  {sorted(outlier_codes)}")

    X = feat[["imd19", "imd_quintile", "mean_no2", "mean_pm25"]].copy()
    X.columns = ["IMD19", "imd_quintile", "mean_no2", "mean_pm25"]
    # Log-transform target: right-skewed prescribing rates
    y = np.log1p(feat["items_per_1000"]).copy()

    print(f"\n[ML] Feature matrix shape: {X.shape}  (rows=practice×month, cols=features)")
    print(f"     Features: {list(X.columns)}")
    print(f"     Target (log items/1000) range: {y.min():.2f} – {y.max():.2f}  "
          f"[raw: {feat['items_per_1000'].min():.1f} – {feat['items_per_1000'].max():.1f}]")
    print(f"     NOTE: NO2/PM25 are synthetic — vary by month, not by practice")

    model = XGBRegressor(
        n_estimators  = 200,
        max_depth     = 3,
        learning_rate = 0.05,
        random_state  = 42,
        verbosity     = 0,
    )

    # ── Train / evaluate ──────────────────────────────────────────────────────
    n = len(X)
    if n < 50:
        print(f"\n[ML] n={n} < 50 — using 5-fold cross-validation")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        mean_r2 = cv_scores.mean()
        results["ml_metric"] = f"CV R²={mean_r2:.3f} (mean of 5 folds, log target)"
        results["ml_r2"]     = round(mean_r2, 3)
        results["ml_method"] = "5-fold CV"
        print(f"     CV R² scores : {[round(s,3) for s in cv_scores]}")
        print(f"     Mean R²      : {mean_r2:.3f}")
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"\n[ML] n={n} >= 50 — 80/20 train/test split (log-transformed target)")
        print(f"     Train: {len(X_train)}  Test: {len(X_test)}")
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        r2       = r2_score(y_test, y_pred)
        # MAE back in original units
        mae_log  = mean_absolute_error(y_test, y_pred)
        mae_raw  = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
        results["ml_metric"] = f"R²={r2:.3f} (log target), MAE={mae_raw:.1f} items/1000"
        results["ml_r2"]     = round(r2, 3)
        results["ml_method"] = "80/20 split"
        print(f"     Test R² (log)     : {r2:.3f}")
        print(f"     Test MAE (raw)    : {mae_raw:.1f} items/1000")
        model.fit(X, y)   # refit on full data for SHAP

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("\n[ML] Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    feature_imp = sorted(zip(X.columns, mean_abs), key=lambda x: x[1], reverse=True)
    top3 = [f[0] for f in feature_imp[:3]]
    results["top_shap"] = top3
    print(f"     SHAP feature importance:")
    for fname, fval in feature_imp:
        print(f"       {fname:<15}  mean|SHAP|={fval:.4f}")

    # Beeswarm
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Beeswarm — Antidepressant Prescribing Rate")
    plt.tight_layout()
    beeswarm_path = PROCESSED / "shap_summary.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved -> {beeswarm_path.relative_to(ROOT)}")

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    bar_path = PROCESSED / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved -> {bar_path.relative_to(ROOT)}")

    # Save model
    model_path = PROCESSED / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"     Model saved -> {model_path.relative_to(ROOT)}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run():
    db = duckdb.connect(str(DB_PATH))
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    df, df_real = load_data(db)

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    a = analysis(df, df_real, db)

    print("\n" + "=" * 60)
    print("MACHINE LEARNING")
    print("=" * 60)
    m = ml(df_real)

    db.close()

    # ── Final summary block ───────────────────────────────────────────────────
    print()
    print("=" * 40)
    print("=== ANALYSIS SUMMARY ===")
    print(f"Antidepressant ratio Q1/Q5   : {a['ad_ratio']:.2f}x  "
          f"(Q1={a['q1_mean']}, Q5={a['q5_mean']} items/1000)")
    print(f"Spearman IMD vs antidep      : r={a['r_imd']:.3f}, p={a['p_imd']:.4f}")
    print(f"Spearman NO2 vs respiratory  : r={a['r_no2']:.3f}, p={a['p_no2']:.4f}  (synthetic AQ)")
    print(f"Outlier practices            : {a['n_outliers']}")
    print(f"ML ({m['ml_method']})         : {m['ml_metric']}")
    print(f"Top SHAP features            : 1. {m['top_shap'][0]}  "
          f"2. {m['top_shap'][1]}  3. {m['top_shap'][2]}")
    print("=" * 40)


if __name__ == "__main__":
    run()
