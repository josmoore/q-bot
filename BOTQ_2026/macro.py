"""Yahoo market context and real-time ALFRED CPI snapshots.

Yahoo symbols match Jeffrey's Sharpe Transformer source (SPY, ^VIX, ^TNX,
^IRX). CPIAUCSL vintage CSVs are fetched from the St. Louis Fed graph endpoint
with ``vintage_date`` set to the final calendar day of each month. Thus each snapshot
contains only CPI observations available by that date; it can update a daily
signal after the CPI release without using later revisions.
"""

from __future__ import annotations

import calendar
import csv
import io
from datetime import date
from urllib.request import urlopen

import polars as pl


YAHOO_SYMBOLS = {"SPY": "spy", "^VIX": "vix", "^TNX": "tnx", "^IRX": "irx"}
ALFRED_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def fetch_yahoo_market(start: str, end: str) -> pl.DataFrame:
    """Download daily market context using the symbols in Jeffrey's notebook."""
    import yfinance as yf

    raw = yf.download(
        list(YAHOO_SYMBOLS), start=start, end=end, auto_adjust=False,
        actions=False, group_by="ticker", progress=False, threads=True,
    )
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no observations for {start} through {end}")
    rows = []
    for symbol, alias in YAHOO_SYMBOLS.items():
        series = raw[symbol]["Close"] if isinstance(raw.columns, __import__("pandas").MultiIndex) else raw["Close"]
        rows.extend(
            {"date": idx.date(), "series": alias, "value": float(value)}
            for idx, value in series.dropna().items()
        )
    return pl.DataFrame(rows).pivot(on="series", index="date", values="value").sort("date")


def fetch_alfred_month_end_vintages(start: date, end: date) -> pl.DataFrame:
    """Download one point-in-time CPIAUCSL snapshot per month from ALFRED.

    ``vintage_date`` is month end, not the observation month. Consumers must
    join the latest vintage whose date is on or before the signal date.
    """
    rows = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        vintage = date(year, month, calendar.monthrange(year, month)[1])
        url = f"{ALFRED_URL}?id=CPIAUCSL&vintage_date={vintage.isoformat()}"
        with urlopen(url, timeout=30) as response:
            csv_text = response.read().decode("utf-8")
        for row in csv.DictReader(io.StringIO(csv_text)):
            value = row[f"CPIAUCSL_{vintage:%Y%m%d}"]
            if value and value != ".":
                if date.fromisoformat(row["observation_date"]) >= vintage.replace(day=1):
                    raise ValueError(f"Unexpected observation in vintage {vintage}: {row}")
                rows.append({
                    "vintage_date": vintage,
                    "observation_date": date.fromisoformat(row["observation_date"]),
                    "cpi": float(value),
                })
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return pl.DataFrame(rows).sort(["vintage_date", "observation_date"])


def vintage_inflation_changes(vintages: pl.DataFrame) -> pl.DataFrame:
    """Calculate the last available month-over-month change in YoY CPI per vintage."""
    required = {"vintage_date", "observation_date", "cpi"}
    if missing := required - set(vintages.columns):
        raise ValueError(f"Missing CPI vintage fields: {sorted(missing)}")
    vintage_rows = (
        vintages.sort(["vintage_date", "observation_date"])
        .with_columns(
            ((pl.col("cpi") / pl.col("cpi").shift(12).over("vintage_date") - 1) * 100)
            .alias("inflation_yoy")
        )
        .with_columns(
            (pl.col("inflation_yoy") - pl.col("inflation_yoy").shift(1).over("vintage_date"))
            .alias("inflation_change")
        )
        .drop_nulls("inflation_change")
        .group_by("vintage_date", maintain_order=True)
        .last()
        .select(pl.col("vintage_date").alias("date"), "inflation_change")
        .sort("date")
    )
    return vintage_rows
