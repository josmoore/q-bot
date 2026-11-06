"""Compute and cache a deterministic shard of daily DMD forecasts."""
import argparse
from datetime import date
from pathlib import Path

import polars as pl

from BOTQ_2026.dmd import walk_forward_stock_signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("shard", type=int)
    parser.add_argument("count", type=int)
    args = parser.parse_args()
    assets = pl.scan_parquet(args.cache / "assets.parquet")
    ids = (
        assets.filter(pl.col("in_universe")).select("barrid").unique()
        .sort("barrid").collect()["barrid"].to_list()[args.shard::args.count]
    )
    source = assets.select("date", "barrid", "price", "in_universe")
    calendar = source.select("date").unique().sort("date").collect()
    prices = source.filter(pl.col("barrid").is_in(ids)).collect().partition_by("barrid")
    rows = []
    for stock in prices:
        aligned = calendar.join(stock, on="date", how="left", validate="1:1").sort("date")
        scores = walk_forward_stock_signal(aligned["price"].to_numpy())
        rows.append(aligned.with_columns(pl.Series("dmd", scores)).filter(
            pl.col("in_universe") & (pl.col("date") >= date(2005, 1, 1))
        ).select(pl.col("date").alias("signal_date"), "barrid", "dmd"))
    output = args.cache / f"dmd-{args.shard:03d}.parquet"
    temp = output.with_suffix(".tmp")
    pl.concat(rows).write_parquet(temp)
    temp.replace(output)


if __name__ == "__main__":
    main()
