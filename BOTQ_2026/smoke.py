"""Small ORC data and optimizer check; no backtest or intensive processing."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import sf_quant.data as sfd

from BOTQ_2026.data import audit_one_date, benchmark_weights, scan_assets
from BOTQ_2026.portfolio import optimize_day


def main() -> None:
    day = date(2010, 1, 4)
    root = Path.home()
    sfd.env(root=str(root), database="production")
    assets = scan_assets(root, "production", day, day)
    print(audit_one_date(assets, day))
    sample = (
        benchmark_weights(assets)
        .sort("benchmark_weight", descending=True)
        .limit(8)
        .collect()
    )
    sample = sample.with_columns(
        pl.Series("combined_alpha", np.linspace(-0.001, 0.001, sample.height))
    ).join(
        assets.select("barrid", "predicted_beta").collect(), on="barrid"
    )
    sample = sample.with_columns(
        (pl.col("benchmark_weight") / pl.col("benchmark_weight").sum()).alias(
            "benchmark_weight"
        )
    )
    for long_only in (False, True):
        result = optimize_day(day, sample, long_only=long_only)
        print(
            result["portfolio"][0],
            "assets=", result.height,
            "active_beta=", float((result["active_weight"] * result["predicted_beta"]).sum()),
            "predicted_te=", result["predicted_tracking_error"][0],
        )


if __name__ == "__main__":
    main()
