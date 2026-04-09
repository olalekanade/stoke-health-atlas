"""
03_ingest_air_quality.py
Attempts to fetch real NO2/PM2.5 data from DEFRA UK-AIR for Stoke station STKS.
Falls back to synthetic seasonal data if requests fail or return empty/HTML.
Saves monthly means to DuckDB table air_quality_monthly.
"""

import io
import random
from pathlib import Path

import requests
import pandas as pd
import duckdb

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "health_atlas.duckdb"

STATIONS   = ["STKS"]
YEARS      = [2023, 2024]
POLLUTANTS = {"NO2": "no2", "PM25": "pm25"}

UKAIR_URL = (
    "https://uk-air.defra.gov.uk/data_files/site_data/{station}_{year}.csv"
)


# ── helpers ────────────────────────────────────────────────────────────────────
def try_fetch_ukair(station: str, year: int) -> pd.DataFrame | None:
    url = UKAIR_URL.format(station=station, year=year)
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {url}")
            return None
        text = resp.text.strip()
        if not text or text.lower().startswith("<!"):
            print(f"    Empty / HTML response for {station} {year}")
            return None
        # UK-AIR CSVs have 4 header rows before the actual data
        df = pd.read_csv(io.StringIO(text), header=4, low_memory=False)
        if df.empty or len(df.columns) < 2:
            print(f"    Parsed CSV has no usable data for {station} {year}")
            return None
        print(f"    OK: {len(df)} rows, columns: {list(df.columns)[:6]}")
        return df
    except Exception as exc:
        print(f"    Error fetching {station} {year}: {exc}")
        return None


def parse_real_data(df: pd.DataFrame, station: str, year: int) -> list[dict]:
    """Extract monthly means for NO2 and PM2.5 from a raw UK-AIR DataFrame."""
    records = []
    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # Find the date column (first column is usually 'Date' or 'date')
    date_col = df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.dropna(subset=[date_col])
        df["_month"] = df[date_col].dt.month
        df["_year"]  = df[date_col].dt.year
    except Exception:
        return records

    pollutant_map = {
        "NO2":  ["NO2", "Nitrogen dioxide", "Nitrogen Dioxide"],
        "PM25": ["PM2.5", "PM25", "Particulate matter < 2.5", "PM2.5 particulate"],
    }

    for label, aliases in pollutant_map.items():
        col = next((c for c in df.columns for a in aliases
                    if a.lower() in c.lower()), None)
        if col is None:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        monthly = (
            df.groupby(["_year", "_month"])[col]
            .mean()
            .reset_index()
        )
        for _, row in monthly.iterrows():
            if pd.notna(row[col]):
                records.append({
                    "station":           station,
                    "pollutant":         label,
                    "year":              int(row["_year"]),
                    "month":             int(row["_month"]),
                    "mean_concentration": round(float(row[col]), 2),
                    "is_synthetic":      False,
                })
    return records


def make_synthetic(years: list[int] | None = None) -> list[dict]:
    """Generate plausible seasonal synthetic data for the given years."""
    if years is None:
        years = [2025, 2026]
    random.seed(42)
    stations = ["Stoke-on-Trent Centre", "Stoke-on-Trent A500"]
    # Seasonal index: higher in winter months
    seasonal = {
        1: 1.20, 2: 1.15, 3: 1.00, 4: 0.92, 5: 0.85, 6: 0.80,
        7: 0.78, 8: 0.80, 9: 0.90, 10: 0.98, 11: 1.12, 12: 1.18,
    }
    base = {"NO2": 28.0, "PM25": 12.5}
    spread = {"NO2": 6.0, "PM25": 3.5}

    records = []
    for station in stations:
        for pollutant, b in base.items():
            for year in years:
                for month, idx in seasonal.items():
                    noise = random.uniform(-0.5, 0.5)
                    value = round(b * idx + spread[pollutant] * (idx - 1) + noise, 2)
                    records.append({
                        "station":            station,
                        "pollutant":          pollutant,
                        "year":               year,
                        "month":              month,
                        "mean_concentration": value,
                        "is_synthetic":       True,
                    })
    return records


# ── main ───────────────────────────────────────────────────────────────────────
def run():
    all_records: list[dict] = []
    used_synthetic = False

    print("Attempting DEFRA UK-AIR fetches...")
    for station in STATIONS:
        for year in YEARS:
            print(f"  {station} {year}")
            raw = try_fetch_ukair(station, year)
            if raw is not None:
                parsed = parse_real_data(raw, station, year)
                print(f"    Parsed {len(parsed)} monthly records")
                all_records.extend(parsed)

    if not all_records:
        print("\nNo real data retrieved — generating synthetic data.")
        all_records = make_synthetic()
        used_synthetic = True
    else:
        print(f"\nReal data: {len(all_records)} monthly records total.")

    df = pd.DataFrame(all_records)
    print(f"\nData shape: {df.shape}")
    print(df.head())

    db = duckdb.connect(str(DB_PATH))
    db.execute("""
        CREATE OR REPLACE TABLE air_quality_monthly AS
        SELECT * FROM df
    """)
    count = db.execute("SELECT COUNT(*) FROM air_quality_monthly").fetchone()[0]
    synth_flag = db.execute(
        "SELECT COUNT(*) FROM air_quality_monthly WHERE is_synthetic"
    ).fetchone()[0]
    db.close()

    print(f"\nSaved to air_quality_monthly: {count} rows "
          f"({'synthetic' if used_synthetic else 'real'}, "
          f"{synth_flag} synthetic rows)")
    print("Done.")


if __name__ == "__main__":
    run()
