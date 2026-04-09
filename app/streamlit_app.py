"""
Stoke-on-Trent Urban Health Atlas — Streamlit app
"""

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
PROC       = ROOT / "data" / "processed"
GPKG_PATH  = PROC / "stoke_lsoa.gpkg"
SHAP_BAR   = PROC / "shap_bar.png"
SHAP_BSWRM = PROC / "shap_summary.png"

METRIC_COLS = {
    "items_per_1000":     "items_per_1000",
    "IMD19":              "imd19",
    "mean_concentration": "mean_no2",
}
METRIC_LABELS = {
    "items_per_1000":     "Items per 1,000 patients",
    "IMD19":              "IMD 2019 national rank",
    "mean_concentration": "NO2 mean concentration (ug/m3)",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stoke-on-Trent Urban Health Atlas",
    layout="wide",
    page_icon="🏥",
)

# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_atlas_monthly() -> pd.DataFrame:
    try:
        return pd.read_parquet(PROC / "atlas_monthly.parquet")
    except Exception as e:
        st.info(f"atlas_monthly not available: {e}")
        return pd.DataFrame()


@st.cache_data
def load_analysis_quintile() -> pd.DataFrame:
    try:
        return pd.read_parquet(PROC / "analysis_quintile.parquet")
    except Exception as e:
        st.info(f"analysis_quintile not available: {e}")
        return pd.DataFrame()


@st.cache_data
def load_gp_practices() -> pd.DataFrame:
    try:
        df = pd.read_parquet(PROC / "gp_practices.parquet")
        return df[["organisation_code", "name"]].sort_values("name")
    except Exception as e:
        st.info(f"gp_practices not available: {e}")
        return pd.DataFrame()


@st.cache_data
def load_lsoa_boundaries() -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(str(GPKG_PATH))
    except Exception as e:
        st.info(f"LSOA boundaries not available: {e}")
        return gpd.GeoDataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏥 Health Atlas")
    st.markdown("**Stoke-on-Trent & Staffordshire ICB**")
    st.divider()

    drug_cat = st.selectbox(
        "Drug category",
        ["antidepressants", "respiratory", "cardiovascular", "diabetes"],
    )
    map_metric = st.selectbox(
        "Map metric",
        ["items_per_1000", "IMD19", "mean_concentration"],
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Practices", "215")
        st.metric("Months of data", "3")
    with col_b:
        st.metric("LSOAs covered", "146")
        st.metric("Prescribing rows", "1,233,564")

    st.caption("NHSBSA EPD · ONS IMD 2019 · DEFRA UK-AIR · NHS ODS")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_atlas    = load_atlas_monthly()
df_quintile = load_analysis_quintile()
df_gp       = load_gp_practices()
gdf         = load_lsoa_boundaries()

st.title("Stoke-on-Trent Urban Health Atlas")
st.markdown(
    f"Showing **{drug_cat.replace('_', ' ').title()}** · "
    f"Map metric: **{map_metric}**"
)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — MAP + DEPRIVATION GRADIENT
# ══════════════════════════════════════════════════════════════════════════════
col_map, col_bar = st.columns([3, 2])

# ── LEFT: Folium choropleth ───────────────────────────────────────────────────
with col_map:
    st.subheader("LSOA-level prescribing map")

    if df_atlas.empty or gdf.empty:
        st.info("Map data not available.")
    else:
        metric_col = METRIC_COLS[map_metric]

        # Aggregate atlas_monthly -> LSOA level for selected drug + metric
        df_filt = df_atlas[df_atlas["drug_category"] == drug_cat].copy()
        if metric_col not in df_filt.columns:
            st.info(f"Column '{metric_col}' not found in atlas_monthly.")
        else:
            df_agg = (
                df_filt.groupby("lsoa21cd")[metric_col]
                .mean()
                .reset_index()
                .rename(columns={metric_col: "metric_value", "lsoa21cd": "LSOA21CD"})
            )
            df_agg["metric_value"] = df_agg["metric_value"].round(2)

            # Merge onto GeoDataFrame for tooltip
            gdf_plot = gdf.merge(df_agg, on="LSOA21CD", how="left")
            gdf_plot["metric_value"] = gdf_plot["metric_value"].fillna(0)
            # Ensure WGS84 for Folium
            if gdf_plot.crs and gdf_plot.crs.to_epsg() != 4326:
                gdf_plot = gdf_plot.to_crs(epsg=4326)

            m = folium.Map(
                location=[53.002, -2.179],
                zoom_start=12,
                tiles="CartoDB positron",
            )

            choropleth = folium.Choropleth(
                geo_data=gdf_plot.__geo_interface__,
                data=df_agg,
                columns=["LSOA21CD", "metric_value"],
                key_on="feature.properties.LSOA21CD",
                fill_color="YlOrRd",
                fill_opacity=0.75,
                line_opacity=0.3,
                nan_fill_color="lightgrey",
                bins=6,
                legend_name=METRIC_LABELS[map_metric],
                name="Choropleth",
            )
            choropleth.add_to(m)

            # Tooltip overlay
            tooltip_gdf = gdf_plot[["LSOA21CD", "LSOA21NM", "metric_value", "geometry"]].copy()
            tooltip_gdf["metric_value"] = tooltip_gdf["metric_value"].astype(str)
            folium.GeoJson(
                tooltip_gdf,
                style_function=lambda f: {
                    "fillOpacity": 0,
                    "color": "transparent",
                    "weight": 0,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["LSOA21CD", "LSOA21NM", "metric_value"],
                    aliases=["LSOA Code", "LSOA Name", METRIC_LABELS[map_metric]],
                    sticky=True,
                ),
            ).add_to(m)

            st_folium(m, width=700, height=480)

# ── RIGHT: Deprivation gradient bar chart ─────────────────────────────────────
with col_bar:
    st.subheader("Deprivation gradient")

    if df_quintile.empty:
        st.info("analysis_quintile table not available.")
    else:
        df_q = (
            df_quintile[df_quintile["drug_category"] == "antidepressants"]
            .groupby("imd_quintile")["avg_rate"]
            .mean()
            .reset_index()
            .sort_values("imd_quintile")
        )

        if df_q.empty:
            st.info("No antidepressant data in analysis_quintile.")
        else:
            bar_colors = ["#8B0000", "#D73027", "#FC8D59", "#91BFDB", "#4575B4"]
            fig_bar = go.Figure()
            for i, row in df_q.iterrows():
                q  = int(row["imd_quintile"])
                v  = row["avg_rate"]
                ci = min(q - 1, len(bar_colors) - 1)
                fig_bar.add_trace(go.Bar(
                    x=[f"Q{q}"],
                    y=[v],
                    marker_color=bar_colors[ci],
                    showlegend=False,
                    name=f"Q{q}",
                ))

            # Annotation on Q1 bar
            q1_val = df_q[df_q["imd_quintile"] == 1]["avg_rate"].values
            if len(q1_val):
                fig_bar.add_annotation(
                    x="Q1", y=float(q1_val[0]),
                    text="1.85× higher than Q5",
                    showarrow=True, arrowhead=2,
                    ax=40, ay=-40,
                    font=dict(size=11, color="#8B0000"),
                )

            fig_bar.update_layout(
                title="Deprivation gradient — antidepressant prescribing",
                xaxis_title="IMD Quintile (1=most deprived)",
                yaxis_title="Antidepressant items per 1,000 patients",
                height=420,
                margin=dict(t=50, b=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig_bar.update_yaxes(gridcolor="#eeeeee")
            st.plotly_chart(fig_bar, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — GP EXPLORER + SHAP
# ══════════════════════════════════════════════════════════════════════════════
col_prac, col_shap = st.columns([1, 1])

# ── LEFT: GP practice explorer ────────────────────────────────────────────────
with col_prac:
    st.subheader("GP practice explorer")

    if df_gp.empty or df_atlas.empty:
        st.info("Practice data not available.")
    else:
        practice_names = df_gp["name"].dropna().sort_values().unique().tolist()
        selected_name  = st.selectbox("Select a practice", practice_names)

        # Map name -> code
        code_row = df_gp[df_gp["name"] == selected_name]
        if not code_row.empty:
            pcode = code_row["organisation_code"].iloc[0]
            df_prac = df_atlas[df_atlas["practice_code"] == pcode].copy()

            if df_prac.empty:
                st.info(f"No atlas data for {selected_name}.")
            else:
                df_trend = (
                    df_prac.groupby(["month", "drug_category"])["items_per_1000"]
                    .mean()
                    .reset_index()
                )
                fig_line = px.line(
                    df_trend,
                    x="month",
                    y="items_per_1000",
                    color="drug_category",
                    markers=True,
                    title=f"Prescribing trend — {selected_name}",
                    labels={
                        "month": "Month",
                        "items_per_1000": "Items per 1,000 patients",
                        "drug_category": "Drug category",
                    },
                )
                fig_line.update_layout(
                    height=400,
                    margin=dict(t=50, b=40),
                    plot_bgcolor="white",
                )
                fig_line.update_xaxes(
                    tickvals=[11, 12, 1],
                    ticktext=["Nov 2025", "Dec 2025", "Jan 2026"],
                )
                st.plotly_chart(fig_line, width="stretch")

# ── RIGHT: Model explainability ───────────────────────────────────────────────
with col_shap:
    st.subheader("What drives prescribing rates?")

    if SHAP_BAR.exists():
        st.image(str(SHAP_BAR), caption="SHAP feature importance (mean |SHAP value|)")
    else:
        st.info("SHAP chart not found — run pipeline/05_analysis_and_ml.py first.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Q1/Q5 ratio", "1.85×")
    m2.metric("Spearman r (IMD)", "−0.365")
    m3.metric("Model R²", "0.50")

    st.caption(
        "XGBoost + SHAP · trained on 128 practices with real IMD scores · "
        "AQ data synthetic"
    )

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — IMD SCATTER
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("IMD deprivation score vs antidepressant prescribing rate")

if df_atlas.empty:
    st.info("atlas_monthly not available.")
else:
    df_scatter = df_atlas[
        (df_atlas["drug_category"] == "antidepressants") &
        (df_atlas["imd19"].notna())
    ].copy()
    df_scatter["imd_quintile_str"] = "Q" + df_scatter["imd_quintile"].astype(int).astype(str)

    if df_scatter.empty:
        st.info("No antidepressant data with IMD19 scores.")
    else:
        try:
            fig_scatter = px.scatter(
                df_scatter,
                x="imd19",
                y="items_per_1000",
                color="imd_quintile_str",
                color_discrete_sequence=px.colors.diverging.RdYlGn[::-1],
                trendline="ols",
                hover_data=["practice_name", "imd19", "items_per_1000"],
                labels={
                    "imd19":           "IMD 2019 national rank",
                    "items_per_1000":  "Antidepressant items per 1,000 patients",
                    "imd_quintile_str":"IMD Quintile",
                },
                title="IMD deprivation score vs antidepressant prescribing rate",
            )
        except Exception:
            # Fallback if statsmodels not available
            fig_scatter = px.scatter(
                df_scatter,
                x="imd19",
                y="items_per_1000",
                color="imd_quintile_str",
                color_discrete_sequence=px.colors.diverging.RdYlGn[::-1],
                hover_data=["practice_name", "imd19", "items_per_1000"],
                labels={
                    "imd19":           "IMD 2019 national rank",
                    "items_per_1000":  "Antidepressant items per 1,000 patients",
                    "imd_quintile_str":"IMD Quintile",
                },
                title="IMD deprivation score vs antidepressant prescribing rate",
            )

        fig_scatter.update_layout(
            height=420,
            plot_bgcolor="white",
            margin=dict(t=50, b=40),
        )
        fig_scatter.update_yaxes(gridcolor="#eeeeee")
        st.plotly_chart(fig_scatter, width="stretch")
        st.caption(
            "Spearman r = −0.365, p = 0.0001 · each point = one practice-month · "
            "Stoke-on-Trent & Staffordshire ICB"
        )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Data: NHSBSA EPD (Nov 2025–Jan 2026) · ONS IMD 2019 · "
    "DEFRA UK-AIR (synthetic) · NHS ODS · Open Government Licence v3.0"
)
