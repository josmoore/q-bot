"""Join cached signal forecasts without using unreleased inflation scores."""
import argparse
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--last-year", type=int, default=2025)
    args = parser.parse_args()
    dmd = [args.cache / f"dmd-{i:03d}.parquet" for i in range(128)]
    transformer = [args.cache / f"transformer-{year}.parquet" for year in range(2005, args.last_year + 1)]
    missing = [path.name for path in dmd + transformer if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} forecast shards: {missing[:8]}")
    inflation = pl.scan_parquet(args.cache / "inflation_beta.parquet").sort("available_date")
    output = args.cache / ("forecasts.parquet" if args.last_year == 2025 else f"forecasts-through-{args.last_year}.parquet")
    temporary = output.with_suffix(".tmp")
    (
        pl.scan_parquet(dmd).filter(pl.col("signal_date").dt.year() <= args.last_year)
        .join(pl.scan_parquet(transformer), on=["signal_date", "barrid"], how="left", validate="1:1")
        .sort("signal_date")
        .join_asof(inflation, left_on="signal_date", right_on="available_date", by="barrid",
                   strategy="backward", tolerance="45d")
        .select("signal_date", "barrid", "dmd", "sharpe_transformer", "inflation_beta")
        .sink_parquet(temporary)
    )
    temporary.replace(output)


if __name__ == "__main__":
    main()
