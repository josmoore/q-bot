"""Attribute DMD's stock-level forecast and forward residual payoff to styles."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

CONTROLS = {
    "barra_momentum": "USSLOWL_MOMENTUM",
    "barra_reversal": "USSLOWL_LTREVRSL",
    "sf_momentum": "momentum",
    "sf_reversal": "reversal",
}
SPECS = {
    **{name: (name,) for name in CONTROLS},
    "momentum": ("barra_momentum", "sf_momentum"),
    "reversal": ("barra_reversal", "sf_reversal"),
    "barra": ("barra_momentum", "barra_reversal"),
    "sf_quant": ("sf_momentum", "sf_reversal"),
    "all": tuple(CONTROLS),
}


def join_style_controls(panel):
    """Join same-signal-date style controls for a single calendar year."""
    import sf_quant.data as sfd

    start, end = panel["signal_date"].min(), panel["signal_date"].max()
    exposures = sfd.load_exposures(start, end, False, [CONTROLS["barra_momentum"], CONTROLS["barra_reversal"]])
    signals = sfd.load_scores(start, end, names=[CONTROLS["sf_momentum"], CONTROLS["sf_reversal"]])
    for source, names, value in (
        (exposures, ("barra_momentum", "barra_reversal"), None),
        (signals, ("sf_momentum", "sf_reversal"), "score"),
    ):
        if value:
            source = source.group_by("date", "barrid").agg(*(
                pl.col(value).filter(pl.col("signal_name") == CONTROLS[name]).first().alias(name)
                for name in names
            ))
        else:
            source = source.rename({CONTROLS[name]: name for name in names})
        panel = panel.join(source.rename({"date": "signal_date"}),
                             on=["signal_date", "barrid"], how="left", validate="m:1")
    return panel


def dmd_style_scores(day):
    """Return residual DMD and raw DMD on identical coverage."""
    dmd = day["dmd"].to_numpy()
    styles = day.select(*CONTROLS).to_numpy()
    matched = np.isfinite(dmd) & np.isfinite(styles).all(axis=1)
    if matched.sum() < 30:
        raise ValueError(f"Insufficient DMD style coverage on {day['signal_date'][0]}")
    dmd_matched = dmd[matched] - dmd[matched].mean()
    x = styles[matched] - styles[matched].mean(axis=0)
    residual = np.zeros(len(dmd))
    residual[matched] = dmd_matched - x @ np.linalg.lstsq(x, dmd_matched, rcond=None)[0]
    raw = np.zeros(len(dmd))
    raw[matched] = dmd_matched
    return residual, raw


def daily_attribution(dmd, y, styles):
    """OLS projection within one signal date. Payoff is exactly additive."""
    dmd = dmd - dmd.mean()
    y = y - y.mean()
    styles = styles - styles.mean(axis=0)
    ss = dmd @ dmd
    gross = np.abs(dmd).sum()
    if ss <= 0:
        return []
    result = []
    raw = (dmd @ y) / gross
    for spec, controls in SPECS.items():
        cols = [list(CONTROLS).index(c) for c in controls]
        x = styles[:, cols]
        fitted = x @ np.linalg.lstsq(x, dmd, rcond=None)[0]
        residual = dmd - fitted
        residual_ss = residual @ residual
        result.append(dict(
            controls=spec, score_r2=1 - residual_ss / ss,
            raw_payoff=raw, explained_payoff=(fitted @ y) / gross,
            remaining_payoff=(residual @ y) / gross,
            raw_slope=dmd @ y / ss,
            adjusted_slope=residual @ y / residual_ss if residual_ss > 1e-12 else None,
            score_correlation=(dmd @ x[:, 0]) / np.sqrt(ss * (x[:, 0] @ x[:, 0]))
            if len(cols) == 1 and np.any(x[:, 0]) else None,
        ))
    return result


def main():
    import sf_quant.data as sfd
    from BOTQ_2026.walk_forward import build_panel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sfd.env(root=str(Path.home()), database="production")
    assets = pl.scan_parquet(args.cache / "assets.parquet").filter(
        pl.col("date").is_between(date(2009, 12, 1), date(2024, 12, 31))
    ).select("date", "barrid", "in_universe", "market_cap", "return", "predicted_beta").collect()
    forecasts = pl.scan_parquet(args.cache / "forecasts-through-2024.parquet").select(
        "signal_date", "barrid", "dmd").collect()
    panel = build_panel(assets, forecasts, pl.read_parquet(args.cache / "exit_return_audit.parquet")).filter(
        pl.col("date") >= date(2010, 1, 1)
    ).select("signal_date", "label_available_date", "barrid", "dmd", "forward_residual_return")
    rows, coverage = [], []
    for year in range(2009, 2025):
        subset = panel.filter(pl.col("signal_date").dt.year() == year)
        start, end = subset["signal_date"].min(), subset["signal_date"].max()
        if start is None:
            continue
        subset = join_style_controls(subset)
        for group in subset.partition_by("signal_date", maintain_order=True):
            dt = group["signal_date"][0]
            eligible = group.filter(pl.col("dmd").is_finite() & pl.col("forward_residual_return").is_finite())
            matched = eligible.filter(pl.all_horizontal(pl.col(c).is_finite() for c in CONTROLS))
            coverage.append(dict(signal_date=dt, eligible=eligible.height, matched=matched.height,
                                 **{name: eligible[name].is_finite().sum() for name in CONTROLS}))
            if matched.height >= 30:
                rows.extend(dict(signal_date=dt, stocks=matched.height, **row)
                            for row in daily_attribution(matched["dmd"].to_numpy(),
                                                        matched["forward_residual_return"].to_numpy(),
                                                        matched.select(*CONTROLS).to_numpy()))
        print(f"{year}: cumulative regression rows {len(rows)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(coverage).sort("signal_date").write_csv(args.output / "coverage.csv")
    if not rows:
        raise ValueError("No overlapping DMD and style score dates; inspect coverage.csv")
    daily = pl.DataFrame(rows).sort("signal_date", "controls")
    daily.write_parquet(args.output / "daily.parquet")
    summary = daily.group_by("controls").agg(
        pl.len().alias("sessions"), pl.col("stocks").mean(),
        *(pl.col(c).mean() for c in ("score_r2", "score_correlation", "raw_payoff",
                                     "explained_payoff", "remaining_payoff", "raw_slope", "adjusted_slope")),
    ).with_columns((pl.col("explained_payoff") / pl.col("raw_payoff")).alias("payoff_share_explained"))
    summary.sort("controls").write_csv(args.output / "summary.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = daily.filter(pl.col("controls") == "all")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in (("raw_payoff", "Raw DMD"), ("explained_payoff", "Style component"),
                       ("remaining_payoff", "DMD after controls")):
        ax.plot(both["signal_date"].to_list(), both[col].cum_sum().to_numpy(), label=label)
    ax.set(ylabel="Cumulative residual payoff per unit of raw DMD gross",
           xlabel="Signal date", title="DMD payoff attribution: Barra and sf-quant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "dmd_payoff_attribution.png", dpi=150)


if __name__ == "__main__":
    main()
