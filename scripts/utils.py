
import pandas as pd
import numpy as np
import math
import os
import sys
import time
from pathlib import Path

def get_tick_size(price):
    """Hyperliquid perp tick: prices keep at most 5 significant figures, integer prices always allowed.
    Ponytail: ignores the (6 - szDecimals) max-decimals cap, which only binds for coins priced well below $1."""
    return min(1.0, 10.0 ** (math.floor(math.log10(price)) - 4))


# Only files written in the last 24 h are read: the 50 x 15-min chunks need 12.5 h, the rest is GARCH history.
# Ponytail: relies on file mtime (collector rotates files every 5 min); if data is copied without
# preserving mtimes, nothing is filtered and run time grows with history again.
LOOKBACK_S = 24 * 3600


def safe_read_parquet(path):
    """
    Safely read parquet file or directory, skipping corrupted files.
    """
    path_obj = Path(path)
    
    if path_obj.is_file():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"Warning: Failed to read {path}. Skipping. Error: {e}")
            return pd.DataFrame()

    elif path_obj.is_dir():
        dfs = []
        cutoff = time.time() - LOOKBACK_S
        files = sorted(p for p in path_obj.glob("*.parquet") if p.stat().st_mtime >= cutoff)
        if not files:
            raise ValueError(f"No parquet files modified in the last {LOOKBACK_S / 3600:.0f} h in {path}")
             
        for p in files:
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                # The file the collector is still writing has no footer yet ("magic bytes not found"): skip quietly
                if "magic bytes" not in str(e).lower():
                    print(f"Warning: Skipping potentially incomplete/corrupted file: {p.name}. Error: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No valid data could be read from {path}")
        
        return pd.concat(dfs, ignore_index=True)

    else:
        raise ValueError(f"Path not found: {path}")


def event_time(df):
    """
    Exchange (block) time in ms where recorded, else local receive time. Trades and book snapshots then share
    one clock, so "the book strictly before a trade" holds even when websocket messages arrive out of order.
    """
    local = pd.to_datetime(df['timestamp'], unit='s')
    if 'exchange_timestamp' not in df.columns:
        return local.astype('datetime64[ns]')
    exchange = pd.to_datetime(pd.to_numeric(df['exchange_timestamp'], errors='coerce'), unit='ms')
    return exchange.fillna(local).astype('datetime64[ns]')


def load_trades_data(parquet_path):
    """Load trades data from a Parquet file/directory."""
    df = safe_read_parquet(parquet_path)
    
    if df.empty:
        raise ValueError(f"Parquet file at {parquet_path} is empty or all files were skipped.")
    
    if 'timestamp' not in df.columns:
        if 'timestamp' in df.index.names:
            df = df.reset_index()
        else:
            raise ValueError(f"Parquet file at {parquet_path} missing 'timestamp' column. Available columns: {df.columns.tolist()}")

    # Remove duplicates based on trade_id if available
    if 'trade_id' in df.columns:
        df = df.drop_duplicates(subset=['trade_id'])
    else:
        df = df.drop_duplicates()

    df['datetime'] = event_time(df)
    df = df.set_index('datetime')
    df = df.sort_index()
    return df


def effective_side_price(df, side, threshold=1000):
    """
    Per snapshot: price of the first level where cumulative notional (price * size) reaches `threshold`,
    scanning at most 20 levels and stopping at the first missing level. Falls back to the best level
    when the visible book is thinner than `threshold`.
    """
    n = 0
    while n < 20 and f'{side}_price_{n}' in df.columns and f'{side}_size_{n}' in df.columns:
        n += 1
    if n == 0:
        raise ValueError(f"No {side} levels in order book data")
    p = df[[f'{side}_price_{i}' for i in range(n)]].to_numpy(float)
    v = p * df[[f'{side}_size_{i}' for i in range(n)]].to_numpy(float)
    ok = np.cumprod(~np.isnan(v), axis=1).astype(bool)          # levels before the first missing one
    hit = ok & (np.cumsum(np.where(ok, v, 0.0), axis=1) >= threshold)
    return np.where(hit.any(axis=1), p[np.arange(len(p)), hit.argmax(axis=1)], p[:, 0])


def load_effective_book(parquet_path, threshold=1000):
    """
    One row per order book snapshot (event time): depth-weighted bid/ask at `threshold` notional and their mid.
    Use this, not the 1-s grid, to find the book state just before a trade.
    """
    df = safe_read_parquet(parquet_path)

    if df.empty:
        raise ValueError(f"Parquet file at {parquet_path} is empty or all files were skipped.")

    if 'timestamp' not in df.columns:
        if 'timestamp' in df.index.names:
            df = df.reset_index()
        else:
            raise ValueError(f"Parquet file at {parquet_path} missing 'timestamp' column.")

    book = pd.DataFrame({'price_bid': effective_side_price(df, 'bid', threshold),
                         'price_ask': effective_side_price(df, 'ask', threshold)},
                        index=pd.DatetimeIndex(event_time(df), name='datetime'))
    book['mid_price'] = (book['price_bid'] + book['price_ask']) / 2
    book = book.dropna().sort_index()

    if book.empty:
        raise ValueError("No valid effective mid-price data after processing.")
    return book


def effective_mid_grid(book, max_stale_s=60):
    """
    1-s grid of the book. label='right', closed='left': the value at time t is the last state strictly before t
    (causal). Seconds more than `max_stale_s` after the last update are dropped instead of forward-filled.
    """
    r = book.resample('s', label='right', closed='left')
    grid = r.last().ffill()
    fresh = r['mid_price'].count() > 0
    stale_s = (~fresh).groupby(fresh.cumsum()).cumcount()
    return grid[stale_s <= max_stale_s].dropna()


def load_effective_mid_price(parquet_path, threshold=1000):
    """Effective mid-price on a 1-s grid (see effective_mid_grid)."""
    return effective_mid_grid(load_effective_book(parquet_path, threshold))
