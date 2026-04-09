"""
02_ingest_reference_data.py
Fetches:
  1. IMD 2019 LSOA data for Stoke-on-Trent  -> DuckDB table imd_lsoa
     (IMD 2025 not yet published on ArcGIS; 2019 is the most recent for England)
  2. LSOA 2021 boundaries for Stoke          -> data/processed/stoke_lsoa.gpkg
     (two-step: lookup LSOA21CD codes, then pull geometries)
  3. NHS ODS GP practices (active, ST*)      -> DuckDB table gp_practices
"""

import io
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import duckdb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
PROCESSED  = ROOT / "data" / "processed"
GP_RAW_DIR = ROOT / "data" / "raw" / "gp_practices"
IMD_DIR    = ROOT / "data" / "raw" / "imd"
DB_PATH    = ROOT / "health_atlas.duckdb"

for d in (PROCESSED, GP_RAW_DIR, IMD_DIR):
    d.mkdir(parents=True, exist_ok=True)

ARCGIS   = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services"
LAD_CODE = "E06000021"   # Stoke-on-Trent


# ── 1. IMD 2019 (most recent England data available) ──────────────────────────
def ingest_imd(db: duckdb.DuckDBPyConnection) -> int:
    print("\n[1/3] Fetching IMD 2019 LSOA data for Stoke-on-Trent...")
    url = f"{ARCGIS}/Index_of_Multiple_Deprivation_Dec_2019_Lookup_in_England_2022/FeatureServer/0/query"
    params = {
        "where":             f"LAD19CD='{LAD_CODE}'",
        "outFields":         "*",
        "f":                 "json",
        "resultRecordCount": 2000,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    features = data.get("features", [])
    if not features:
        raise RuntimeError(f"No IMD features returned: {data.get('error')}")

    records = [f["attributes"] for f in features]
    df = pd.DataFrame(records)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df):,}")

    db.execute("CREATE OR REPLACE TABLE imd_lsoa AS SELECT * FROM df")
    count = db.execute("SELECT COUNT(*) FROM imd_lsoa").fetchone()[0]
    print(f"  Saved to imd_lsoa: {count:,} rows")
    return count


# ── 2. LSOA 2021 Boundaries ────────────────────────────────────────────────────
def ingest_boundaries() -> int:
    out = PROCESSED / "stoke_lsoa.gpkg"
    print("\n[2/3] Fetching LSOA 2021 boundaries for Stoke-on-Trent...")

    # Step A: look up LSOA21CD codes that belong to Stoke
    lookup_url = (
        f"{ARCGIS}/LSOA_2021_to_Ward_to_Lower_Tier_Local_Authority_May_2022"
        f"_Lookup_for_England_2022/FeatureServer/0/query"
    )
    r = requests.get(lookup_url, params={
        "where":             f"LTLA22CD='{LAD_CODE}'",
        "outFields":         "LSOA21CD",
        "f":                 "json",
        "resultRecordCount": 2000,
    }, timeout=60)
    r.raise_for_status()
    lsoa_codes = [f["attributes"]["LSOA21CD"] for f in r.json().get("features", [])]
    print(f"  LSOA21CD codes for Stoke: {len(lsoa_codes)}")
    if not lsoa_codes:
        raise RuntimeError("No LSOA codes returned from lookup service.")

    # Step B: fetch geometries via POST (IN clause with 163 codes exceeds GET URL limit)
    codes_str = ",".join(f"'{c}'" for c in lsoa_codes)
    boundary_url = f"{ARCGIS}/LSOA_2021_EW_BSC_V4_RUC/FeatureServer/0/query"
    r2 = requests.post(boundary_url, data={
        "where":             f"LSOA21CD IN ({codes_str})",
        "outFields":         "LSOA21CD,LSOA21NM",
        "f":                 "geojson",
        "resultRecordCount": 2000,
    }, timeout=60)
    r2.raise_for_status()

    gdf = gpd.read_file(io.StringIO(r2.text))
    print(f"  CRS: {gdf.crs}  |  Features: {len(gdf):,}")
    gdf.to_file(out, driver="GPKG")
    print(f"  Saved -> {out.relative_to(ROOT)}")
    return len(gdf)


# ── 3. GP Practices — extracted from prescribing_raw ──────────────────────────
# The NHS ODS epraccur.zip (files.digital.nhs.uk) returns 403 when accessed
# programmatically. Instead we derive unique practices directly from the
# prescribing data already loaded into DuckDB, which contains PRACTICE_CODE,
# PRACTICE_NAME, ADDRESS_1-4, and POSTCODE for every QNC/Staffordshire practice.

def ingest_gp_practices(db: duckdb.DuckDBPyConnection) -> int:
    print("\n[3/3] Extracting GP practices from prescribing_raw...")

    # Check prescribing_raw exists
    tables = [r[0] for r in db.execute("SHOW TABLES").fetchall()]
    if "prescribing_raw" not in tables:
        raise RuntimeError(
            "prescribing_raw not found in DuckDB — run 01_ingest_prescribing.py first."
        )

    db.execute("""
        CREATE OR REPLACE TABLE gp_practices AS
        SELECT
            PRACTICE_CODE                           AS organisation_code,
            MAX(PRACTICE_NAME)                      AS name,
            MAX(ADDRESS_1)                          AS address_1,
            MAX(ADDRESS_2)                          AS address_2,
            MAX(ADDRESS_3)                          AS address_3,
            MAX(ADDRESS_4)                          AS address_4,
            MAX(POSTCODE)                           AS postcode,
            MAX(ICB_NAME)                           AS icb_name,
            MAX(ICB_CODE)                           AS icb_code,
            MAX(PCO_NAME)                           AS pco_name,
            COUNT(*)                                AS prescribing_rows
        FROM prescribing_raw
        WHERE PRACTICE_CODE IS NOT NULL
        GROUP BY PRACTICE_CODE
        ORDER BY PRACTICE_CODE
    """)

    count = db.execute("SELECT COUNT(*) FROM gp_practices").fetchone()[0]
    # Also save a CSV for inspection
    df = db.execute("SELECT * FROM gp_practices").df()
    df.to_csv(GP_RAW_DIR / "gp_practices_from_prescribing.csv", index=False)
    print(f"  Unique practices extracted: {count:,}")
    print(f"  Saved to gp_practices table and {GP_RAW_DIR.name}/gp_practices_from_prescribing.csv")
    return count


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    db = duckdb.connect(str(DB_PATH))

    imd_count  = ingest_imd(db)
    lsoa_count = ingest_boundaries()
    gp_count   = ingest_gp_practices(db)

    db.close()

    print("\n" + "=" * 52)
    print("Reference data summary")
    print("=" * 52)
    print(f"  imd_lsoa        (DuckDB table) : {imd_count:>6,} rows")
    print(f"  stoke_lsoa.gpkg (GeoPackage)   : {lsoa_count:>6,} features")
    print(f"  gp_practices    (DuckDB table) : {gp_count:>6,} rows")
    print("Done.")


if __name__ == "__main__":
    run()
