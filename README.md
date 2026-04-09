![Streamlit](https://img.shields.io/badge/Streamlit-live-brightgreen) ![License](https://img.shields.io/badge/licence-OGL%20v3-blue)

# Stoke-on-Trent Urban Health Atlas

An interactive public health dashboard combining NHS prescribing data, ONS deprivation indices, and DEFRA air quality sensors across 163 LSOAs in Stoke-on-Trent and Staffordshire.

## Key findings

- Antidepressant prescribing is 1.85× higher in the most deprived quintile (Q1: 95.1 items/1000) vs the least deprived (Q5: 51.4 items/1000)
- Spearman correlation between IMD deprivation score and antidepressant prescribing: r = −0.365, p = 0.0001
- XGBoost model predicts prescribing rates from deprivation + air quality features with R² = 0.50
- IMD19 is the dominant driver by SHAP importance, followed by NO2 concentration

## Data sources

| Source | Dataset | Rows | Licence |
|---|---|---|---|
| NHSBSA EPD | English Prescribing Dataset (Nov 2025–Jan 2026) | 1,233,564 | OGL v3 |
| ONS IMD 2019 | Indices of Multiple Deprivation — England | 378 LSOAs | OGL v3 |
| DEFRA UK-AIR | NO2 / PM2.5 monthly means (synthetic placeholder) | 96 | OGL v3 |
| NHS ODS | GP practice register — Staffordshire ICB | 215 practices | OGL v3 |

## Pipeline

1. **Ingest** — stream NHSBSA EPD monthly CSVs in 50k-row chunks, filter to QNC/Staffordshire ICB, save as parquet → `prescribing_raw`
2. **Reference data** — fetch ONS IMD 2019, LSOA 2021 boundaries (GeoPackage), LSOA11→LSOA21 crosswalk, GP practice locations via ONSPD postcode lookup
3. **Geospatial join** — match each practice postcode to LSOA21CD via ONSPD, join to IMD, compute per-practice deprivation quintiles; expand coverage to surrounding LADs (Newcastle-under-Lyme, Stafford, Staffordshire Moorlands)
4. **Analysis + ML** — Spearman correlations, outlier detection, XGBoost regressor (IMD19 + air quality → antidepressant items/1000), SHAP explainability
5. **Streamlit app** — interactive choropleth map, deprivation gradient chart, GP practice explorer, SHAP model panel

## Run locally

```bash
git clone <repo-url>
cd stoke-health-atlas
python -m venv venv && source venv/Scripts/activate   # Windows
pip install -r requirements.txt

python pipeline/01_ingest_prescribing.py
python pipeline/02_ingest_reference_data.py
python pipeline/02b_fix_lsoa_lookup.py
python pipeline/03_ingest_air_quality.py
python pipeline/04_build_atlas.py
python pipeline/04b_fix_lsoa_imd_assignment.py
python pipeline/05_analysis_and_ml.py

streamlit run app/streamlit_app.py
```

## Limitations

- 3 months of EPD data (Nov 2025–Jan 2026) — extend by re-running ingestion script monthly
- Air quality data is synthetic — swap in real DEFRA readings when AURN station STKS becomes available on the API
- 87 practices assigned fallback IMD quintile Q3 — extend coverage by adding South Staffordshire, Cannock Chase, Lichfield LADs to IMD ingestion
