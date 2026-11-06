"""Fetch small external macro inputs on a login node (not on compute nodes).

Usage: ``python -m BOTQ_2026.fetch_macro --out data/macro``. Yahoo data is
downloaded once for Jeffrey's SPY/VIX/TNX/IRX symbols. CPI is fetched from
ALFRED as month-end vintages so historic observations reflect the values
available at that vintage, including later revisions only after publication.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from BOTQ_2026.macro import fetch_alfred_month_end_vintages, fetch_yahoo_market


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    end_exclusive = args.end + timedelta(days=1)
    yahoo = fetch_yahoo_market(args.start.isoformat(), end_exclusive.isoformat())
    cpi = fetch_alfred_month_end_vintages(args.start, args.end)
    yahoo.write_parquet(args.out / "yahoo_market.parquet", compression="zstd")
    cpi.write_parquet(args.out / "cpi_alfred_vintages.parquet", compression="zstd")
    print(f"Yahoo market rows: {yahoo.height}; CPI vintage rows: {cpi.height}")


if __name__ == "__main__":
    main()
