"""Build monthly inflation forecasts using vintages known at calendar month end."""
import argparse
from datetime import date
from pathlib import Path

import polars as pl

from BOTQ_2026.macro import vintage_inflation_changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    cpi = vintage_inflation_changes(pl.read_parquet(args.cache / "cpi_alfred_vintages.parquet"))
    monthly = (
        pl.scan_parquet(args.cache / "assets.parquet")
        .select(pl.col("date").dt.month_end(), "barrid", "return")
        .group_by("date", "barrid")
        # sf-quant asset returns are percentage points; compounding needs decimals.
        .agg(((1 + pl.col("return") / 100).product() - 1).alias("stock_return"),
             pl.col("return").count().alias("n_returns"))
        .filter(pl.col("n_returns") > 0)
        .join(cpi.lazy(), on="date")
        .sort("date")
        .collect()
    )
    # Calendar-based windows ensure gaps in listing history do not extend the lookback.
    result = (
        monthly.with_columns((pl.col("stock_return") * pl.col("inflation_change")).alias("xy"),
                             pl.col("inflation_change").pow(2).alias("yy"))
        .rolling("date", period="60mo", group_by="barrid")
        .agg(pl.len().alias("n"), pl.col("stock_return").mean().alias("x"),
             pl.col("inflation_change").mean().alias("y"),
             pl.col("xy").mean(), pl.col("yy").mean())
        .with_columns((pl.col("yy") - pl.col("y").pow(2)).alias("variance"))
        .filter((pl.col("n") >= 36) & (pl.col("variance") > 0))
        .select(pl.col("date").alias("available_date"), "barrid",
                (-(pl.col("xy") - pl.col("x") * pl.col("y")) / pl.col("variance")).alias("inflation_beta"))
    )
    result.write_parquet(args.cache / "inflation_beta.parquet")
    print(f"Saved {result.height:,} monthly scores", flush=True)


if __name__ == "__main__":
    main()
