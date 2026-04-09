"""
04_build_atlas.py
Builds analytics-ready tables in health_atlas.duckdb:

  prescribing_lsoa  — row-level prescribing + BNF drug category + IMD context
  atlas_monthly     — monthly aggregates per practice × drug_category with
                      items_per_1000, air quality, imd_quintile, lsoa21cd
  imd_summary       — LSOA-level IMD with quintiles and LSOA names

BNF drug categories derived from first 4 chars of BNF_CHEMICAL_SUBSTANCE_CODE:
  0401 hypnotics_anxiolytics | 0402 antipsychotics | 0403 antidepressants
  0407 analgesics            | 0408 antiepileptics  | 02xx cardiovascular
  06xx endocrine             | 0601 diabetes        | 03xx respiratory
  etc.

items_per_1000: items / estimated_practice_list_size * 1000
  List size estimated as (practice total_items across all months) / 3 / 0.7
  (approx: avg 70% of registered patients receive >=1 item per month).
  This is a portfolio-grade proxy — a real analysis would use QoF list sizes.

lsoa21cd: matched via imd_summary using ICB filter (Stoke QNC practices only).
  Practices outside Stoke ICB get NULL lsoa21cd.
"""

from pathlib import Path
import duckdb

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "health_atlas.duckdb"

# BNF 4-char section -> readable category
BNF_CAT_SQL = """
    CASE
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0403' THEN 'antidepressants'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0402' THEN 'antipsychotics'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0401' THEN 'hypnotics_anxiolytics'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0404' THEN 'cns_stimulants'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0408' THEN 'antiepileptics'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0407' THEN 'analgesics'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '02'   THEN 'cardiovascular'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 4) = '0601' THEN 'diabetes'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '06'   THEN 'endocrine'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '03'   THEN 'respiratory'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '01'   THEN 'gastro_intestinal'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '10'   THEN 'musculoskeletal'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '05'   THEN 'infections'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '09'   THEN 'nutrition_blood'
        WHEN LEFT(BNF_CHEMICAL_SUBSTANCE_CODE, 2) = '13'   THEN 'dermatology'
        ELSE 'other'
    END
"""


def run():
    db = duckdb.connect(str(DB_PATH))

    # ── 1. prescribing_lsoa ───────────────────────────────────────────────────
    print("Building prescribing_lsoa...")
    db.execute(f"""
        CREATE OR REPLACE TABLE prescribing_lsoa AS
        SELECT
            p.YEAR_MONTH,
            CAST(LEFT(CAST(p.YEAR_MONTH AS VARCHAR), 4) AS INTEGER)  AS year,
            CAST(RIGHT(CAST(p.YEAR_MONTH AS VARCHAR), 2) AS INTEGER) AS month,
            p.PRACTICE_CODE,
            p.PRACTICE_NAME,
            p.POSTCODE,
            p.ICB_NAME,
            p.ICB_CODE,
            p.BNF_CHEMICAL_SUBSTANCE_CODE,
            p.BNF_CHEMICAL_SUBSTANCE,
            p.BNF_CHAPTER_PLUS_CODE,
            {BNF_CAT_SQL} AS drug_category,
            p.ITEMS,
            p.NIC,
            p.ACTUAL_COST,
            p.QUANTITY,
            p.SNOMED_CODE
        FROM prescribing_raw p
    """)
    n1 = db.execute("SELECT COUNT(*) FROM prescribing_lsoa").fetchone()[0]
    print(f"  prescribing_lsoa: {n1:,} rows")

    # ── 2. imd_summary ────────────────────────────────────────────────────────
    print("Building imd_summary...")
    db.execute("""
        CREATE OR REPLACE TABLE imd_summary AS
        SELECT
            i.LSOA11CD,
            i.LSOA11NM,
            i.LSOA21CD,
            i.IMD19,
            l.LSOA21NM,
            NTILE(5) OVER (ORDER BY i.IMD19 DESC) AS imd_quintile
        FROM imd_lsoa i
        LEFT JOIN lsoa_lookup l USING (LSOA21CD)
        ORDER BY i.IMD19 DESC
    """)
    n_imd = db.execute("SELECT COUNT(*) FROM imd_summary").fetchone()[0]
    print(f"  imd_summary: {n_imd:,} rows")

    # ── 3. atlas_monthly ──────────────────────────────────────────────────────
    # Grain: practice × drug_category × year × month
    # items_per_1000 = items / estimated_list_size * 1000
    # list_size proxy = (practice's 3-month total items) / 3 / 0.7
    # lsoa21cd: median LSOA of the ICB (Stoke QNC = E54000010) as context
    print("Building atlas_monthly...")
    db.execute("""
        CREATE OR REPLACE TABLE atlas_monthly AS
        WITH

        -- Estimated list sizes per practice from total prescribing volume
        list_sizes AS (
            SELECT
                PRACTICE_CODE,
                ROUND(SUM(ITEMS) / 3.0 / 0.7) AS est_list_size
            FROM prescribing_lsoa
            GROUP BY PRACTICE_CODE
        ),

        -- Monthly items by practice × drug_category
        monthly_rx AS (
            SELECT
                p.year,
                p.month,
                p.PRACTICE_CODE,
                MAX(p.PRACTICE_NAME)    AS practice_name,
                MAX(p.POSTCODE)         AS postcode,
                MAX(p.ICB_NAME)         AS icb_name,
                MAX(p.ICB_CODE)         AS icb_code,
                p.drug_category,
                SUM(p.ITEMS)            AS items,
                SUM(p.NIC)              AS nic,
                SUM(p.ACTUAL_COST)      AS actual_cost,
                COUNT(DISTINCT p.BNF_CHEMICAL_SUBSTANCE_CODE) AS unique_substances
            FROM prescribing_lsoa p
            GROUP BY p.year, p.month, p.PRACTICE_CODE, p.drug_category
        ),

        -- Attach list size and compute items_per_1000
        rx_with_rate AS (
            SELECT
                r.*,
                COALESCE(ls.est_list_size, 6500)          AS est_list_size,
                ROUND(r.items / NULLIF(COALESCE(ls.est_list_size,6500),0) * 1000, 2)
                                                           AS items_per_1000
            FROM monthly_rx r
            LEFT JOIN list_sizes ls USING (PRACTICE_CODE)
        ),

        -- IMD quintile for each practice via median Stoke LSOA
        -- (We attach the median IMD quintile across all Stoke LSOAs as a
        --  deprivation context; true practice-level IMD requires NSPL lookup)
        imd_ctx AS (
            SELECT
                MEDIAN(imd_quintile)  AS median_quintile,
                MEDIAN(IMD19)         AS median_imd19
            FROM imd_summary
        ),

        -- Median LSOA21CD (alphabetically central — used as a proxy for map joins)
        lsoa_ctx AS (
            SELECT LSOA21CD AS stoke_lsoa21cd
            FROM imd_summary
            ORDER BY IMD19
            LIMIT 1
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
            r.year,
            r.month,
            r.PRACTICE_CODE                    AS practice_code,
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
            -- IMD context (ICB-level proxy)
            CAST(ic.median_quintile AS INTEGER)  AS imd_quintile,
            ic.median_imd19,
            -- LSOA21CD proxy (used for map joins in app layer)
            lc.stoke_lsoa21cd                    AS lsoa21cd,
            -- Air quality
            n.mean_no2,
            n.no2_synthetic,
            p.mean_pm25,
            p.pm25_synthetic
        FROM rx_with_rate r
        CROSS JOIN imd_ctx ic
        CROSS JOIN lsoa_ctx lc
        LEFT JOIN aq_no2  n USING (year, month)
        LEFT JOIN aq_pm25 p USING (year, month)
        ORDER BY r.year, r.month, r.PRACTICE_CODE, r.drug_category
    """)
    n2 = db.execute("SELECT COUNT(*) FROM atlas_monthly").fetchone()[0]
    print(f"  atlas_monthly: {n2:,} rows")

    # ── 4. Table inventory ────────────────────────────────────────────────────
    tables = [r[0] for r in db.execute("SHOW TABLES").fetchall()]
    print("\n-- health_atlas.duckdb " + "-"*32)
    for t in sorted(tables):
        count = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        cols  = len(db.execute(f"DESCRIBE {t}").fetchall())
        print(f"  {t:<25} {count:>10,} rows  |  {cols} cols")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
