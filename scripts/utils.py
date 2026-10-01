"""Market data, UTC clocks and atomic JSON publication. No trading side effects."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("A timezone-aware UTC timestamp is required")
    return stamp.tz_convert("UTC")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def get_tick_size(price, sz_decimals=0):
    if not math.isfinite(price) or price <= 0 or not 0 <= sz_decimals <= 6:
        raise ValueError("Invalid price or size precision")
    return max(10.0 ** -(6 - sz_decimals),
               min(1.0, 10.0 ** (math.floor(math.log10(price)) - 4)))


def round_price(price, sz_decimals, buy):
    tick = get_tick_size(price, sz_decimals)
    result = (math.floor(price / tick + 1e-9) if buy else math.ceil(price / tick - 1e-9)) * tick
    return round(result, 6 - sz_decimals)


def _intersects(pf, start, end):
    """Parquet event-time statistics prune reads; mtime never defines the sample."""
    names = pf.schema_arrow.names
    column = "exchange_timestamp" if "exchange_timestamp" in names else "timestamp"
    if column not in names:
        return True
    idx = names.index(column)
    scale = 1000 if column == "exchange_timestamp" else 1
    for i in range(pf.num_row_groups):
        stat = pf.metadata.row_group(i).column(idx).statistics
        if stat is None or not stat.has_min_max:
            return True
        try:
            if (start is None or stat.max >= start.timestamp() * scale) and (
                end is None or stat.min <= end.timestamp() * scale
            ):
                return True
        except TypeError:
            return True
    return False


def safe_read_parquet(path, start=None, end=None, columns=None):
    path = Path(path)
    start, end = (utc(start) if start is not None else None), (utc(end) if end is not None else None)
    files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
    frames, corrupt = [], []
    for part in files:
        try:
            pf = pq.ParquetFile(part)
            if _intersects(pf, start, end):
                selected = None if columns is None else [c for c in columns if c in pf.schema_arrow.names]
                frames.append(pf.read(columns=selected).to_pandas())
        except Exception as exc:
            corrupt.append(f"{part.name}: {exc}")
    if corrupt:
        raise ValueError("Unreadable published parquet: " + "; ".join(corrupt[:3]))
    if not frames:
        raise ValueError(f"No published observations in requested interval: {path}")
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" not in df and df.index.name == "timestamp":
        df = df.reset_index()
    if "timestamp" not in df:
        raise ValueError("Missing local receipt timestamp")
    received = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    event = pd.to_datetime(pd.to_numeric(df.get("exchange_timestamp"), errors="coerce"),
                           unit="ms", utc=True, errors="coerce")
    if not isinstance(event, pd.Series):
        event = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    df["clock_valid"] = event.notna() & received.notna() & ((event - received).dt.total_seconds() <= 1)
    df["event_time"] = event.fillna(received)
    df["received_at"] = received
    df["available_at"] = pd.concat([df["event_time"], received], axis=1).max(axis=1)
    df = df.dropna(subset=["event_time", "received_at"])
    if start is not None:
        df = df[df.event_time >= start]
    if end is not None:
        df = df[(df.event_time <= end) & (df.available_at <= end)]
    return df.sort_values(["event_time", "received_at"], kind="stable").reset_index(drop=True)


def stream_path(root, ticker, kind):
    root = Path(root)
    native = root / f"{kind}_{ticker}.parquet"
    return native if native.exists() else root / ticker / kind


def load_trades_data(path, start=None, end=None):
    df = safe_read_parquet(path, start, end)
    required = {"price", "size", "side", "trade_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Trade data missing {sorted(required - set(df.columns))}")
    good = np.isfinite(df.price) & np.isfinite(df["size"]) & (df.price > 0) & (df["size"] > 0)
    good &= df.side.isin(["buy", "sell"]) & df.trade_id.notna() & ~df.trade_id.astype(str).isin(["None", ""])
    if not good.all():
        raise ValueError(f"Invalid trade observations: {int((~good).sum())}")
    keys = ["event_time", "trade_id"] + (["symbol"] if "symbol" in df else [])
    return df.drop_duplicates(keys).set_index("event_time").sort_index()


def load_book_data(path, start=None, end=None, depth=20):
    columns = ["timestamp", "exchange_timestamp", "symbol"] + [
        f"{side}_{field}_{i}" for i in range(depth) for side in ("bid", "ask") for field in ("price", "size")]
    df = safe_read_parquet(path, start, end, columns)
    required = ["bid_price_0", "ask_price_0", "bid_size_0", "ask_size_0"]
    if not set(required).issubset(df.columns):
        raise ValueError("Missing top-of-book prices/sizes")
    values = df[required].to_numpy(float)
    good = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1) & (df.bid_price_0 < df.ask_price_0)
    if not good.all():
        raise ValueError(f"Invalid/crossed book observations: {int((~good).sum())}")
    for side, sign in (("bid", -1), ("ask", 1)):
        cols = [f"{side}_price_{i}" for i in range(20) if f"{side}_price_{i}" in df]
        p = df[cols].to_numpy(float)
        sizes = df[[name.replace("price", "size") for name in cols]].to_numpy(float)
        present = ~np.isnan(p)
        if np.any(present != ~np.isnan(sizes)) or np.any(present & (~np.isfinite(p) | ~np.isfinite(sizes) | (p <= 0) | (sizes <= 0))):
            raise ValueError("Invalid visible depth")
        if np.any(present & ~np.cumprod(present, axis=1).astype(bool)):
            raise ValueError("Hole inside visible depth")
        if np.any(sign * np.diff(p, axis=1) < 0):
            raise ValueError("Unsorted depth")
    df["mid_price"] = (df.bid_price_0 + df.ask_price_0) / 2
    return df.drop_duplicates(["event_time", "received_at"]).set_index("event_time").sort_index()


def load_funding_data(path, start=None, end=None):
    df = safe_read_parquet(path, start, end)
    if not np.isfinite(df.funding_rate).all():
        raise ValueError("Invalid settled funding rates")
    keys = ["event_time"] + (["symbol"] if "symbol" in df else [])
    if (df.groupby(keys).funding_rate.nunique() > 1).any():
        raise ValueError("Conflicting funding observations")
    return df.drop_duplicates(keys).set_index("event_time").sort_index()


def mid_grid(book, start, end, seconds=5, max_age=2):
    """Only snapshots received strictly before a decision can supply its mid."""
    grid = pd.DataFrame({"time": pd.date_range(utc(start), utc(end), freq=f"{seconds}s")})
    available = book.reset_index().sort_values("available_at")
    aligned = pd.merge_asof(grid, available[["available_at", "event_time", "mid_price", "clock_valid"]],
                            left_on="time", right_on="available_at", direction="backward",
                            allow_exact_matches=False, tolerance=pd.Timedelta(seconds=max_age))
    good = (aligned.time - aligned.event_time).dt.total_seconds().between(0, max_age)
    aligned.loc[~good | ~aligned.clock_valid.eq(True), "mid_price"] = np.nan
    return aligned.set_index("time")["mid_price"]
