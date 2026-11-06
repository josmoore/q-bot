"""Sharpe Transformer pieces adapted from Jeffrey's Sharpe_Transformer.ipynb.

Stock inputs use Russell's point-in-time names and CRSP daily open/ask-high/
bid-low/close/volume. The 12 stock features and six market features match
Jeffrey's source; the model emits one raw cross-sectional alpha score per
stock. Training and scoring must mask each date to that date's Russell
membership. Only dates before the next-close decision may enter its inputs.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl


STOCK_FEATURES = (
    "ret_1", "ret_5", "ret_21", "ret_63", "ret_252", "vol_21", "vol_63",
    "vol_surprise", "range", "dist_52w_high", "mom_rank_21", "mom_rank_252",
)
MARKET_FEATURES = ("spy_ret_1", "spy_ret_21", "vix", "vix_chg_5", "tnx", "tnx_chg_21")


def build_stock_features(panel: pl.LazyFrame) -> pl.LazyFrame:
    """Compute source notebook features using Polars, without future values."""
    x = panel.sort(["barrid", "date"]).with_columns(
        pl.col("price").log().alias("log_close"),
        ((pl.col("askhi") - pl.col("bidlo")) / pl.col("price")).alias("range"),
    )
    for n in (1, 5, 21, 63, 252):
        x = x.with_columns(
            (pl.col("log_close") - pl.col("log_close").shift(n).over("barrid")).alias(f"ret_{n}")
        )
    x = x.with_columns(
        (pl.col("ret_1").rolling_std(21).over("barrid")).alias("vol_21"),
        (pl.col("ret_1").rolling_std(63).over("barrid")).alias("vol_63"),
        (pl.col("daily_volume").clip(lower_bound=1).log()
         - pl.col("daily_volume").clip(lower_bound=1).rolling_mean(63).over("barrid").log())
        .alias("vol_surprise"),
        (pl.col("price") / pl.col("price").rolling_max(252).over("barrid") - 1).alias("dist_52w_high"),
    )
    return x.with_columns(
        (pl.col("ret_21").rank().over("date") / pl.len().over("date")).alias("mom_rank_21"),
        (pl.col("ret_252").rank().over("date") / pl.len().over("date")).alias("mom_rank_252"),
    )


def build_market_features(market: pl.DataFrame) -> pl.DataFrame:
    """Transform Yahoo daily closes for SPY, VIX and 10-year Treasury (^TNX)."""
    required = {"date", "spy", "vix", "tnx"}
    if missing := required - set(market.columns):
        raise ValueError(f"Missing Yahoo market columns: {sorted(missing)}")
    return (
        market.sort("date")
        .with_columns(
            pl.col("spy").log().diff().alias("spy_ret_1"),
            (pl.col("spy").log() - pl.col("spy").log().shift(21)).alias("spy_ret_21"),
            (pl.col("vix") - pl.col("vix").shift(5)).alias("vix_chg_5"),
            (pl.col("tnx") - pl.col("tnx").shift(21)).alias("tnx_chg_21"),
        )
        .select("date", *MARKET_FEATURES)
    )


def cross_sectional_zscores(panel: pl.DataFrame) -> pl.DataFrame:
    """Z-score each daily feature over currently present dated Russell members."""
    return panel.with_columns(
        [
            ((pl.col(c) - pl.col(c).mean().over("date")) /
             (pl.col(c).std().over("date") + 1e-12)).clip(-5, 5).alias(c)
            for c in STOCK_FEATURES
        ]
    )


def make_model():
    """Create Jeffrey-style Transformer; import torch only in a GPU job."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, d=256, heads=8, ff=1024, dropout=0.1):
            super().__init__()
            self.heads = heads
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3*d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, length, d = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(b, length, 3, self.heads, d//self.heads).permute(2,0,3,1,4)
            attn = F.scaled_dot_product_attention(q, k, v)
            x = x + self.drop(self.proj(attn.transpose(1,2).reshape(b,length,d)))
            return x + self.drop(self.ff(self.ln2(x)))

    class SharpeTransformer(nn.Module):
        def __init__(self, lookback=40, sectors=12, d=256, layers=6, heads=8, ff=1024):
            super().__init__()
            n_features = len(STOCK_FEATURES) + len(MARKET_FEATURES)
            self.inp = nn.Linear(n_features, d)
            self.cls = nn.Parameter(torch.randn(1,1,d)*0.02)
            self.pos = nn.Parameter(torch.randn(1,lookback+1,d)*0.02)
            self.sector = nn.Embedding(sectors,d)
            self.blocks = nn.ModuleList(Block(d,heads,ff) for _ in range(layers))
            self.norm, self.head = nn.LayerNorm(d), nn.Linear(d,1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, x, sector_id):
            b = x.shape[0]
            x = torch.cat([self.cls.expand(b,-1,-1)+self.sector(sector_id)[:,None],self.inp(x)],1)+self.pos
            for block in self.blocks: x = block(x)
            return self.head(self.norm(x[:,0])).squeeze(-1)
    return SharpeTransformer()
