"""
02b_fix_lsoa_lookup.py
Downloads the ONS LSOA 2011 -> 2021 lookup for the 159 Stoke IMD rows,
adds LSOA21CD to the imd_lsoa DuckDB table, and reports match quality.

Source: LSOA11_LSOA21_LAD22_EW_LU_v2 (ONS Open Geography ArcGIS, public)
"""

from pathlib import Path
import requests
import pandas as pd
import duckdb

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "health_atlas.duckdb"

LOOKUP_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
    "LSOA11_LSOA21_LAD22_EW_LU_v2/FeatureServer/0/query"
)


def run():
    db = duckdb.connect(str(DB_PATH))

    # ── 1. Get the LSOA11CD codes already in imd_lsoa ────────────────────────
    imd_codes = [r[0] for r in db.execute("SELECT LSOA11CD FROM imd_lsoa").fetchall()]
    print(f"IMD rows to match: {len(imd_codes)}")

    # ── 2. Query lookup via POST (IN clause too long for GET) ─────────────────
    codes_str = ",".join(f"'{c}'" for c in imd_codes)
    print("Fetching LSOA11->LSOA21 lookup from ONS ArcGIS...")
    resp = requests.post(LOOKUP_URL, data={
        "where":             f"LSOA11CD IN ({codes_str})",
        "outFields":         "LSOA11CD,LSOA21CD",
        "f":                 "json",
        "resultRecordCount": 2000,
    }, timeout=60)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    print(f"Lookup rows returned: {len(features)}")

    lookup_df = pd.DataFrame([f["attributes"] for f in features])

    # ── 3. Join LSOA21CD into imd_lsoa ───────────────────────────────────────
    db.execute("CREATE OR REPLACE TABLE _lsoa_lookup AS SELECT LSOA11CD, LSOA21CD FROM lookup_df")

    db.execute("""
        CREATE OR REPLACE TABLE imd_lsoa AS
        SELECT
            i.FID, i.LSOA11CD, i.LSOA11NM,
            i.LAD19CD, i.LAD19NM, i.IMD19,
            l.LSOA21CD
        FROM imd_lsoa i
        LEFT JOIN _lsoa_lookup l USING (LSOA11CD)
    """)
    db.execute("DROP TABLE _lsoa_lookup")

    # ── 4. Report match quality ───────────────────────────────────────────────
    total   = db.execute("SELECT COUNT(*) FROM imd_lsoa").fetchone()[0]
    matched = db.execute("SELECT COUNT(*) FROM imd_lsoa WHERE LSOA21CD IS NOT NULL").fetchone()[0]
    unmatched = total - matched

    print(f"\n-- imd_lsoa after join {'-'*30}")
    print(f"  Total rows       : {total:>3}")
    print(f"  LSOA21CD matched : {matched:>3}")
    print(f"  Unmatched        : {unmatched:>3}")

    if unmatched > 0:
        missing = db.execute(
            "SELECT LSOA11CD FROM imd_lsoa WHERE LSOA21CD IS NULL"
        ).fetchall()
        print(f"\n  Unmatched LSOA11CD codes:")
        for row in missing:
            print(f"    {row[0]}")

    # Print updated schema
    cols = [r[0] for r in db.execute("DESCRIBE imd_lsoa").fetchall()]
    print(f"\n  Columns: {cols}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
