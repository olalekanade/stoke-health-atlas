"""
04b_fix_lsoa_imd_assignment.py
Fixes the ICB-median LSOA/IMD proxy in atlas_monthly with real per-practice values:

  1. Fetch NSPL postcode->LSOA21CD for all 215 practice postcodes via ONSPD_NOV_2025_UK
  2. Join to gp_practices -> practice_code: lsoa21cd
  3. Join lsoa21cd -> imd_lsoa -> IMD19 score per practice
  4. Compute imd_quintile via pandas qcut across the Stoke practice distribution
  5. Rebuild atlas_monthly with per-practice lsoa21cd and imd_quintile
  6. Re-run validation query: antidepressant items_per_1000 by imd_quintile
"""

from pathlib import Path
import requests
import pandas as pd
import duckdb

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "health_atlas.duckdb"

ONSPD_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "ONSPD_NOV_2025_UK/FeatureServer/0/query"
)


# ── 1. Fetch NSPL postcode -> LSOA21CD ────────────────────────────────────────
def fetch_postcode_lsoa(postcodes: list[str]) -> pd.DataFrame:
    print(f"Fetching LSOA21CD for {len(postcodes)} practice postcodes from ONSPD...")
    # Normalise: strip spaces, uppercase (ONSPD stores e.g. 'ST5 1QG')
    norm = [p.strip().upper() for p in postcodes if isinstance(p, str) and p.strip()]
    codes_str = ",".join(f"'{c}'" for c in norm)

    resp = requests.post(ONSPD_URL, data={
        "where":             f"pcds IN ({codes_str})",
        "outFields":         "pcds,lsoa21cd",
        "f":                 "json",
        "resultRecordCount": 2000,
    }, timeout=60)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise RuntimeError(f"No features returned: {resp.json().get('error')}")

    df = pd.DataFrame([f["attributes"] for f in features])
    df.columns = ["postcode", "lsoa21cd"]
    df["postcode"] = df["postcode"].str.strip().str.upper()
    print(f"  ONSPD returned {len(df)} postcode matches")
    return df


# ── 2 & 3. Build practice -> LSOA21CD + IMD19 ────────────────────────────────
def build_practice_imd(db: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # Pull practice postcodes
    gp = db.execute(
        "SELECT organisation_code, postcode FROM gp_practices"
    ).df()
    gp["postcode"] = gp["postcode"].str.strip().str.upper()

    # Fetch NSPL lookup
    nspl = fetch_postcode_lsoa(gp["postcode"].dropna().unique().tolist())

    # Join practice -> lsoa21cd
    practice_lsoa = gp.merge(nspl, on="postcode", how="left")
    matched   = practice_lsoa["lsoa21cd"].notna().sum()
    unmatched = practice_lsoa["lsoa21cd"].isna().sum()
    print(f"\n  Practice LSOA matches  : {matched} / {len(practice_lsoa)}")
    print(f"  Unmatched (null LSOA)  : {unmatched}")
    if unmatched > 0:
        nulls = practice_lsoa[practice_lsoa["lsoa21cd"].isna()]["postcode"].tolist()
        print(f"  Unmatched postcodes    : {nulls}")

    # Join lsoa21cd -> IMD19
    imd = db.execute("SELECT LSOA21CD AS lsoa21cd, IMD19 FROM imd_lsoa").df()
    practice_imd = practice_lsoa.merge(imd, on="lsoa21cd", how="left")

    imd_matched = practice_imd["IMD19"].notna().sum()
    print(f"  Practices with IMD19   : {imd_matched} / {len(practice_imd)}")

    # ── 4. Compute imd_quintile via qcut across practice distribution
    # Use only practices with a real IMD score; assign quintile 3 (median) to nulls
    has_imd = practice_imd["IMD19"].notna()
    practice_imd.loc[has_imd, "imd_quintile"] = pd.qcut(
        practice_imd.loc[has_imd, "IMD19"],
        q=5,
        labels=[1, 2, 3, 4, 5],   # 1 = most deprived (lowest IMD rank), 5 = least
    ).astype(float)
    practice_imd["imd_quintile"] = (
        practice_imd["imd_quintile"].fillna(3).astype(int)
    )

    dist = practice_imd["imd_quintile"].value_counts().sort_index()
    print(f"\n  imd_quintile distribution:")
    for q, n in dist.items():
        print(f"    Q{q}: {n} practices")

    return practice_imd[["organisation_code", "lsoa21cd", "IMD19", "imd_quintile"]]


# ── 5. Rebuild atlas_monthly ──────────────────────────────────────────────────
def rebuild_atlas_monthly(db: duckdb.DuckDBPyConnection,
                          practice_imd: pd.DataFrame) -> int:
    print("\nRebuilding atlas_monthly with per-practice LSOA21CD and imd_quintile...")

    # Load practice lookup into DuckDB
    db.execute("CREATE OR REPLACE TABLE _practice_imd AS SELECT * FROM practice_imd")

    db.execute("""
        CREATE OR REPLACE TABLE atlas_monthly AS
        WITH

        list_sizes AS (
            SELECT PRACTICE_CODE,
                   ROUND(SUM(ITEMS) / 3.0 / 0.7) AS est_list_size
            FROM prescribing_lsoa
            GROUP BY PRACTICE_CODE
        ),

        monthly_rx AS (
            SELECT
                p.year, p.month,
                p.PRACTICE_CODE,
                MAX(p.PRACTICE_NAME)   AS practice_name,
                MAX(p.POSTCODE)        AS postcode,
                MAX(p.ICB_NAME)        AS icb_name,
                MAX(p.ICB_CODE)        AS icb_code,
                p.drug_category,
                SUM(p.ITEMS)           AS items,
                SUM(p.NIC)             AS nic,
                SUM(p.ACTUAL_COST)     AS actual_cost,
                COUNT(DISTINCT p.BNF_CHEMICAL_SUBSTANCE_CODE) AS unique_substances
            FROM prescribing_lsoa p
            GROUP BY p.year, p.month, p.PRACTICE_CODE, p.drug_category
        ),

        rx_with_rate AS (
            SELECT
                r.*,
                COALESCE(ls.est_list_size, 6500)   AS est_list_size,
                ROUND(r.items
                    / NULLIF(COALESCE(ls.est_list_size, 6500), 0)
                    * 1000, 2)                       AS items_per_1000
            FROM monthly_rx r
            LEFT JOIN list_sizes ls USING (PRACTICE_CODE)
        ),

        aq_no2 AS (
            SELECT year, month,
                   AVG(mean_concentration) AS mean_no2,
                   MAX(is_synthetic)        AS no2_synthetic
            FROM air_quality_monthly WHERE pollutant = 'NO2'
            GROUP BY year, month
        ),
        aq_pm25 AS (
            SELECT year, month,
                   AVG(mean_concentration) AS mean_pm25,
                   MAX(is_synthetic)        AS pm25_synthetic
            FROM air_quality_monthly WHERE pollutant = 'PM25'
            GROUP BY year, month
        )

        SELECT
            r.year, r.month,
            r.PRACTICE_CODE   AS practice_code,
            r.practice_name,
            r.postcode,
            r.icb_name,
            r.icb_code,
            r.drug_category,
            r.items,
            r.nic,
            r.actual_cost,
            r.unique_substances,
            r.est_list_size,
            r.items_per_1000,
            -- Per-practice real values (from NSPL + IMD join)
            pi.lsoa21cd,
            pi.IMD19          AS imd19,
            pi.imd_quintile,
            -- Air quality
            n.mean_no2,
            n.no2_synthetic,
            p.mean_pm25,
            p.pm25_synthetic
        FROM rx_with_rate r
        LEFT JOIN _practice_imd pi
               ON r.PRACTICE_CODE = pi.organisation_code
        LEFT JOIN aq_no2  n USING (year, month)
        LEFT JOIN aq_pm25 p USING (year, month)
        ORDER BY r.year, r.month, r.PRACTICE_CODE, r.drug_category
    """)

    db.execute("DROP TABLE _practice_imd")
    count = db.execute("SELECT COUNT(*) FROM atlas_monthly").fetchone()[0]
    lsoa_nulls = db.execute(
        "SELECT COUNT(*) FROM atlas_monthly WHERE lsoa21cd IS NULL"
    ).fetchone()[0]
    print(f"  atlas_monthly rebuilt: {count:,} rows, {lsoa_nulls} null lsoa21cd")
    return count


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    db = duckdb.connect(str(DB_PATH))

    practice_imd = build_practice_imd(db)
    rebuild_atlas_monthly(db, practice_imd)

    # ── 6. Validation query ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validation: antidepressant items_per_1000 by imd_quintile")
    print("(Q1=most deprived, Q5=least deprived)")
    print("=" * 60)
    rows = db.execute("""
        SELECT imd_quintile,
               COUNT(DISTINCT practice_code)          AS practices,
               ROUND(AVG(items_per_1000), 1)          AS avg_rate,
               ROUND(MIN(items_per_1000), 1)          AS min_rate,
               ROUND(MAX(items_per_1000), 1)          AS max_rate
        FROM atlas_monthly
        WHERE drug_category = 'antidepressants'
        GROUP BY imd_quintile
        ORDER BY imd_quintile
    """).fetchall()
    print(f"  {'quintile':>8}  {'practices':>9}  {'avg_rate':>9}  "
          f"{'min_rate':>9}  {'max_rate':>9}")
    print("  " + "-" * 52)
    for r in rows:
        print(f"  {str(r[0]):>8}  {r[1]:>9}  {r[2]:>9}  {r[3]:>9}  {r[4]:>9}")

    # Quick count check
    r3 = db.execute("""
        SELECT COUNT(DISTINCT practice_code) AS practices,
               COUNT(DISTINCT lsoa21cd)      AS lsoas,
               COUNT(DISTINCT month)         AS months
        FROM atlas_monthly
    """).fetchone()
    print(f"\n  Distinct practices : {r3[0]}")
    print(f"  Distinct lsoa21cds : {r3[1]}")
    print(f"  Distinct months    : {r3[2]}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
