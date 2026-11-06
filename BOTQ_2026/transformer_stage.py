"""Fit Jeffrey's Sharpe objective on past sessions and score one calendar year."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.checkpoint import checkpoint

from BOTQ_2026.transformer import MARKET_FEATURES, STOCK_FEATURES, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("year", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    year, cache = args.year, args.cache
    cutoff = date(year, 1, 1)
    first = date(year - 5, 1, 1)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda")
    features = (*STOCK_FEATURES, *MARKET_FEATURES)
    assets = pl.scan_parquet(cache / "assets.parquet")
    calendar = (
        assets.select("date").unique().sort("date").collect()
        .with_columns(pl.col("date").shift(-2).alias("label_date"))
        .with_row_index("session")
    )
    panel = (
        pl.scan_parquet([cache / f"transformer-features-{y}.parquet"
                         for y in range(max(2000, year - 6), year + 1)])
        .join(calendar.lazy(), on="date", validate="m:1")
        .join(assets.select(pl.col("date").alias("label_date"), "barrid",
                            pl.col("return").alias("target")),
              on=["label_date", "barrid"], how="left", validate="m:1")
        .collect()
    )
    ids = panel["barrid"].unique().sort().to_list()
    panel = panel.join(pl.DataFrame({"barrid": ids}).with_row_index("stock"),
                       on="barrid", validate="m:1")
    panel = panel.with_columns((pl.col("stock").cast(pl.Int64) * calendar.height
                               + pl.col("session")).alias("key")).sort("key")
    keys = panel["key"].to_numpy()
    values = panel.select(features).to_numpy().astype(np.float32)
    # Market feature normalization uses only the training period and one row per day.
    training_market = panel.filter(pl.col("date").is_between(first, cutoff, closed="left")).unique("date")
    market = training_market.select(MARKET_FEATURES).to_numpy()
    values[:, -len(MARKET_FEATURES):] -= np.nanmean(market, axis=0)
    values[:, -len(MARKET_FEATURES):] /= np.maximum(np.nanstd(market, axis=0), 1e-6)
    train = panel.filter((pl.col("date") >= first) & (pl.col("label_date") < cutoff)
                         & pl.col("target").is_finite()).partition_by("date", maintain_order=False)
    test = panel.filter(pl.col("date").dt.year() == year).partition_by("date")
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def sequence(day):
        wanted = day["key"].to_numpy()[:, None] - np.arange(39, -1, -1)
        indices = np.searchsorted(keys, wanted).clip(max=len(keys) - 1)
        batch = values[indices]
        valid = (keys[indices] == wanted).all(1) & np.isfinite(batch).all(axis=(1, 2))
        valid &= day["session"].to_numpy() >= 39
        return torch.as_tensor(batch[valid], device=device), day.filter(pl.Series(valid))

    def scores(batch, training):
        result = []
        for chunk in batch.split(256):
            sector = torch.zeros(len(chunk), dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                score = checkpoint(model, chunk, sector, use_reentrant=False) if training else model(chunk, sector)
            result.append(score.float())
        return torch.cat(result)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        if args.pilot:
            order = order[:8]
        for offset in range(0, len(order), 8):
            realized, concentration = [], []
            for i in order[offset:offset + 8]:
                batch, day = sequence(train[i])
                if not len(batch):
                    continue
                weight = scores(batch, True).softmax(0)
                target = torch.as_tensor(day["target"].to_numpy(), device=device, dtype=torch.float32)
                realized.append(weight @ target)
                concentration.append(weight.square().sum())
            if len(realized) < 2:
                continue
            daily = torch.stack(realized)
            loss = -np.sqrt(252) * daily.mean() / (daily.std() + 1e-6)
            loss = loss + 2.0 * torch.stack(concentration).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"{year} epoch {epoch + 1}", flush=True)

    model.eval()
    results = []
    with torch.no_grad():
        for day in test[:5] if args.pilot else test:
            batch, eligible = sequence(day)
            if len(batch):
                results.append(eligible.select(pl.col("date").alias("signal_date"), "barrid")
                               .with_columns(pl.Series("sharpe_transformer", scores(batch, False).cpu().numpy())))
    result = pl.concat(results)
    stem = "transformer-pilot" if args.pilot else "transformer"
    output = cache / f"{stem}-{year}.parquet"
    temp = output.with_suffix(".tmp")
    result.write_parquet(temp)
    temp.replace(output)
    print(f"{year}: saved {result.height:,} forecasts", flush=True)


if __name__ == "__main__":
    main()
