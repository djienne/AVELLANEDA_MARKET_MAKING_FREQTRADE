"""
Runnable checks for the parameter pipeline: `python scripts/check_pipeline.py` (prints OK or raises).
Each check compares against something known: a brute-force reference, a hand-built case, or synthetic truth.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import effective_mid_grid, effective_side_price, get_tick_size, load_effective_book, safe_read_parquet


def check_effective_price_matches_reference():
    """Vectorized depth-weighted price == the original level-by-level loop, incl. NaN tails and thin books."""
    rng = np.random.default_rng(1)
    n, L, thr = 2000, 20, 1000.0
    p = 100.0 - 0.1 * np.arange(L) + np.zeros((n, L))
    s = rng.uniform(0, 5, (n, L)) * rng.choice([1.0, 0.01], (n, 1))      # 1% notional -> book never reaches thr
    cut = rng.integers(0, L + 1, n)                                        # levels >= cut are missing
    for i, c in enumerate(cut):
        (p if i % 2 else s)[i, c:] = np.nan
    df = pd.DataFrame({**{f'bid_price_{i}': p[:, i] for i in range(L)}, **{f'bid_size_{i}': s[:, i] for i in range(L)}})

    def reference(prices, sizes):
        cum = 0.0
        for pr, sz in zip(prices, sizes):
            if np.isnan(pr) or np.isnan(sz):
                break
            cum += pr * sz
            if cum >= thr:
                return pr
        return prices[0]

    expected = np.array([reference(p[i], s[i]) for i in range(n)])
    np.testing.assert_array_equal(effective_side_price(df, 'bid', thr), expected)


def check_tick_size():
    """Hyperliquid: 5 significant figures, integer prices always allowed."""
    for price, tick in [(4249.35, 0.1), (2837.25, 0.1), (98765.0, 1.0), (150.2, 0.01), (0.2134, 1e-5)]:
        assert np.isclose(get_tick_size(price), tick), (price, get_tick_size(price))


def check_retention_skips_old_files(tmp):
    d = Path(tmp, 'retention.parquet')
    d.mkdir()
    pd.DataFrame({'timestamp': [1.0]}).to_parquet(d / 'old.parquet')
    pd.DataFrame({'timestamp': [2.0, 3.0]}).to_parquet(d / 'new.parquet')
    old = time.time() - 25 * 3600
    os.utime(d / 'old.parquet', (old, old))
    assert len(safe_read_parquet(d)) == 2


def check_book_is_causal(tmp):
    """
    Book updates at exchange times 99.5 / 100.7 / 101.4 s (mids 10 / 11 / 12); the first one arrives late locally.
    A trade in the same block as the 100.7 update must see mid 10 (the book before it), as intensity.py merges.
    The 1-s grid at t must hold the last state strictly before t.
    """
    d = Path(tmp, 'orderbooks_X.parquet')
    d.mkdir()
    t0 = 1_700_000_000.0
    mids = [10.0, 11.0, 12.0]
    pd.DataFrame({'timestamp': [t0 + 100.9, t0 + 100.8, t0 + 101.5],              # local receive, out of order
                  'exchange_timestamp': [int((t0 + x) * 1000) for x in (99.5, 100.7, 101.4)],
                  'bid_price_0': [m - 0.05 for m in mids], 'bid_size_0': [1e4] * 3,
                  'ask_price_0': [m + 0.05 for m in mids], 'ask_size_0': [1e4] * 3}).to_parquet(d / 'part.parquet')
    book = load_effective_book(d)
    assert list(book['mid_price']) == mids
    trade = pd.DataFrame({'datetime': pd.to_datetime([int((t0 + 100.7) * 1000)], unit='ms').astype('datetime64[ns]')})
    seen = pd.merge_asof(trade, book.reset_index()[['datetime', 'mid_price']], on='datetime',
                         direction='backward', allow_exact_matches=False)
    assert seen['mid_price'].iloc[0] == 10.0, seen
    grid = effective_mid_grid(book)['mid_price']
    at = lambda x: grid.loc[pd.Timestamp(t0 + x, unit='s')]
    assert (at(100), at(101), at(102)) == (10.0, 11.0, 12.0), grid


if __name__ == '__main__':
    check_effective_price_matches_reference()
    check_tick_size()
    with tempfile.TemporaryDirectory() as tmp:
        check_retention_skips_old_files(tmp)
        check_book_is_causal(tmp)
    print(f'OK (pandas {pd.__version__})')
