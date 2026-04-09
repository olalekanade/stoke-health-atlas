"""
01_ingest_prescribing.py
Streams the 3 most-recent NHS BSA EPD monthly CSVs in chunks, filters rows for
Staffordshire / QNC ICB on the fly, saves parquet per month, and loads all
parquets into health_atlas.duckdb as prescribing_raw.
"""

import io
import re
import time
from pathlib import Path

import requests
import pandas as pd
import duckdb
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "prescribing"
DB_PATH = ROOT / "health_atlas.duckdb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_URL = (
    "https://opendata.nhsbsa.net/api/3/action/"
    "package_show?id=english-prescribing-dataset-epd-with-snomed-code"
)
MONTHS_TO_FETCH = 3
CHUNK_SIZE      = 50_000
SLEEP_BETWEEN   = 2          # seconds between month requests
PROGRESS_EVERY  = 10         # print a line every N chunks


# ── helpers ────────────────────────────────────────────────────────────────────
def fetch_resource_urls() -> list[dict]:
    print("Fetching resource list from NHSBSA CKAN API...")
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN API error: {payload}")
    resources = payload["result"]["resources"]
    csv_res   = [r for r in resources if r.get("format", "").upper() == "CSV"]
    csv_res.sort(key=lambda r: r.get("name", ""), reverse=True)
    print(f"  Found {len(csv_res)} CSV resources total.")
    return csv_res


def icb_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain ICB or COMMISSIONER (case-insensitive)."""
    return [c for c in df.columns if re.search(r"ICB|COMMISSIONER", c, re.IGNORECASE)]


def filter_chunk(chunk: pd.DataFrame, filter_cols: list[str]) -> pd.DataFrame:
    if not filter_cols:
        return chunk.iloc[0:0]          # empty frame — no matching columns
    mask = pd.Series(False, index=chunk.index)
    for col in filter_cols:
        lower = chunk[col].astype(str).str.lower()
        mask |= lower.str.contains("staffordshire", na=False)
        mask |= lower.str.contains("qnc",           na=False)
    return chunk[mask]


# ── per-month streaming ingest ─────────────────────────────────────────────────
def ingest_one(url: str, name: str) -> Path | None:
    safe  = re.sub(r"[^\w]", "_", name)
    out   = RAW_DIR / f"{safe}.parquet"

    if out.exists():
        print(f"  [skip] {name} already cached.")
        return out

    print(f"\n  Streaming: {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    matched_frames: list[pd.DataFrame] = []
    filter_cols:    list[str]           = []
    chunk_num   = 0
    total_rows  = 0

    # Wrap the raw byte stream so pandas can chunk-read it
    reader = pd.read_csv(
        io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace"),
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        chunk_num += 1

        # Detect filter columns from the first chunk
        if chunk_num == 1:
            filter_cols = icb_columns(chunk)
            if filter_cols:
                print(f"  Filter columns: {filter_cols}")
            else:
                print("  [warn] No ICB/COMMISSIONER columns found — skipping month.")
                resp.close()
                return None

        filtered = filter_chunk(chunk, filter_cols)
        if not filtered.empty:
            matched_frames.append(filtered)
            total_rows += len(filtered)

        if chunk_num % PROGRESS_EVERY == 0:
            print(f"    chunk {chunk_num:>5}  |  matched so far: {total_rows:>8,}")

    resp.close()
    print(f"  Done — {chunk_num} chunks read, {total_rows:,} matching rows.")

    if not matched_frames:
        print("  [warn] Zero matching rows — nothing to save.")
        return None

    df_final = pd.concat(matched_frames, ignore_index=True)
    df_final.to_parquet(out, index=False)
    print(f"  Saved -> {out.relative_to(ROOT)}")
    return out


# ── main ───────────────────────────────────────────────────────────────────────
def run():
    resources = fetch_resource_urls()
    recent    = resources[:MONTHS_TO_FETCH]
    names     = [r.get("name", "?") for r in recent]
    print(f"\nProcessing {len(recent)} months: {names}\n")

    parquet_paths: list[Path] = []

    for i, resource in enumerate(recent):
        url  = resource.get("url") or resource.get("download_url", "")
        name = resource.get("name", f"month_{i}")
        path = ingest_one(url, name)
        if path:
            parquet_paths.append(path)
        if i < len(recent) - 1:
            print(f"\n  Sleeping {SLEEP_BETWEEN}s before next request...")
            time.sleep(SLEEP_BETWEEN)

    # ── Load all parquets into DuckDB ──────────────────────────────────────────
    print(f"\nLoading {len(parquet_paths)} parquet(s) into {DB_PATH.name}...")
    if not parquet_paths:
        print("No data to load.")
        return

    glob_pat = str(RAW_DIR / "*.parquet").replace("\\", "/")
    db = duckdb.connect(str(DB_PATH))
    db.execute(f"""
        CREATE OR REPLACE TABLE prescribing_raw AS
        SELECT * FROM read_parquet('{glob_pat}')
    """)

    cols  = [row[0] for row in db.execute("DESCRIBE prescribing_raw").fetchall()]
    count = db.execute("SELECT COUNT(*) FROM prescribing_raw").fetchone()[0]
    db.close()

    print("\n-- prescribing_raw " + "-" * 45)
    print(f"Columns ({len(cols)}):")
    for c in cols:
        print(f"  {c}")
    print(f"\nTotal rows: {count:,}")
    print("\nDone.")


if __name__ == "__main__":
    run()
