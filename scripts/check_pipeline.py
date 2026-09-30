"""
Runnable checks for the parameter pipeline: `python scripts/check_pipeline.py` (prints OK or raises).
Each check compares against something known: a brute-force reference, a hand-built case, or synthetic truth.
"""
import ast
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import MAKER_FEE, half_spreads, optimize_params, simulate
from intensity import calculate_intensity_params
from utils import (effective_mid_grid, effective_side_price, get_tick_size, load_effective_book, load_trades_data,
                   safe_read_parquet)
from volatility import calculate_volatility

ROOT = Path(__file__).resolve().parent.parent


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


def check_backtest_quotes_equal_strategy_quotes():
    """The simulation must quote exactly like the deployed strategy (its function is exec'd from source:
    importing the strategy module needs freqtrade)."""
    src = (ROOT / 'user_data/strategies/avellaneda.py').read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_optimal_spreads')
    env = {'np': np, 'mm_logger': logging.getLogger('check')}
    exec(compile(ast.Module([fn], []), 'avellaneda.py', 'exec'), env)
    mid, sigma, kb, ka, gamma, T = 3000.0, 0.02, 0.7, 0.6, 0.05, 0.25 / 24
    r_b, r_a = env['calculate_optimal_spreads'](mid, sigma, kb, ka, gamma, T, 0.0, MAKER_FEE)
    hb, ha = half_spreads(gamma, mid, sigma, kb, ka, T)
    assert np.isclose(mid - hb, r_b) and np.isclose(mid + ha, r_a), (mid - hb, r_b, mid + ha, r_a)


def check_simulate_hand_cases():
    f, nan = MAKER_FEE, np.nan
    s, bid, ask = np.full(4, 100.0), np.full(4, 99.9), np.full(4, 100.1)
    pnl, q, n = simulate(s, bid, ask, np.full(4, nan), np.full(4, nan))          # no trades
    assert n == 0 and not pnl.any()
    pnl, q, n = simulate(s, bid, ask, np.full(4, 99.0), np.full(4, 101.0))       # both sides cross every step
    assert n == 4 and list(q) == [1, 0, 1, 0]                                     # one fill per step, alternating
    pnl, q, n = simulate(s[:2], bid[:2], ask[:2], np.array([99.8, nan]), np.array([100.3, 100.3]))
    assert n == 2 and np.isclose(pnl[-1], 100.1 * (1 - f) - 99.9 * (1 + f))      # entry, then exit next step
    pnl, q, n = simulate(s[:1], bid[:1], ask[:1], np.array([99.9]), np.array([nan]))
    assert n == 0                                                                 # touching is not through


def check_optimizer_known_limits():
    """Constant mid, uninformed takers 20 bp away both sides: tight quotes earn spread minus fees -> enabled.
    No trades at all -> zero edge -> disabled."""
    periods = list(pd.date_range('2026-01-01', periods=6, freq='15min'))
    t = pd.date_range(periods[0], periods[-1] + pd.Timedelta('15min'), freq='s', inclusive='left')
    mid = pd.DataFrame({'mid_price': 100.0}, index=t)
    tt = t[::15] + pd.Timedelta('5s')
    buys = pd.DataFrame({'price': 100.2}, index=tt[::2])
    sells = pd.DataFrame({'price': 99.8}, index=tt[1::2])
    args = (periods, 0.25, [0.002] * 6, [10.0] * 6, [10.0] * 6, 3, mid)
    with contextlib.redirect_stdout(io.StringIO()):
        gamma, enabled, diag = optimize_params(*args, buys, sells)
    assert enabled and max(diag['edge_full']) > 0, diag
    empty = pd.DataFrame({'price': []}, index=pd.DatetimeIndex([]))
    with contextlib.redirect_stdout(io.StringIO()):
        gamma, enabled, diag = optimize_params(*args, empty, empty)
    assert not enabled and max(diag['fills']) == 0


def write_synthetic(d, end, hours=3.0, seed=0):
    """Collector-layout parquet with known truth: random-walk mid, sigma_daily 0.025, tick 0.1, one-tick spread;
    per side 0.8 taker trades/s whose depth beyond the touch is geometric(0.3) ticks -> k = -ln(0.7)/0.1 per $."""
    rng = np.random.default_rng(seed)
    n, tick, S0, sig = int(hours * 3600), 0.1, 3000.0, 0.025
    ts = end - n + np.arange(n)
    mid = S0 + 0.05 + np.cumsum(np.round(rng.normal(0, S0 * sig / np.sqrt(86400), n) / tick) * tick)
    book = {'timestamp': ts + 0.3, 'exchange_timestamp': (ts * 1000).astype('int64')}   # local = exchange + 0.3 s
    for i in range(20):
        book[f'bid_price_{i}'], book[f'bid_size_{i}'] = mid - 0.05 - tick * i, np.full(n, 2.0)
        book[f'ask_price_{i}'], book[f'ask_size_{i}'] = mid + 0.05 + tick * i, np.full(n, 2.0)
    trades = []
    for side, sgn in (('buy', 1), ('sell', -1)):
        k = rng.poisson(0.8 * n)
        tt = np.sort(rng.uniform(0, n, k))
        trades.append(pd.DataFrame({'timestamp': ts[0] + tt + 0.3, 'exchange_timestamp': ((ts[0] + tt) * 1000).astype('int64'),
                                    'price': mid[tt.astype(int)] + sgn * (0.05 + tick * (rng.geometric(0.3, k) - 1)),
                                    'size': 0.1, 'side': side, 'trade_id': [f'{side}{j}' for j in range(k)]}))
    for name, df in (('orderbooks_ETH.parquet', pd.DataFrame(book)), ('trades_ETH.parquet', pd.concat(trades))):
        Path(d, name).mkdir(parents=True)
        df.to_parquet(Path(d, name, 'part.parquet'), index=False)
    return -np.log(0.7) / tick, 0.8, sig


def check_known_truth_and_calculator(tmp):
    now = time.time()
    k_true, A_true, sig_true = write_synthetic(Path(tmp, 'fresh'), now)
    book = load_effective_book(Path(tmp, 'fresh/orderbooks_ETH.parquet'))
    mid, tr = effective_mid_grid(book), load_trades_data(Path(tmp, 'fresh/trades_ETH.parquet'))
    periods = list(pd.date_range(end=mid.index.max() - pd.Timedelta('15min'), periods=8, freq='15min'))
    with contextlib.redirect_stdout(io.StringIO()):
        sig = np.nanmedian(calculate_volatility(mid, 0.25, periods))
        A, kb, _, ka = calculate_intensity_params(periods, 0.25, tr[tr.side == 'buy'], tr[tr.side == 'sell'],
                                                  np.arange(0.1, 4.95, 0.1), book)
    for est, true in ((sig, sig_true), (np.nanmedian(kb), k_true), (np.nanmedian(ka), k_true), (np.nanmedian(A), A_true)):
        assert abs(est / true - 1) < 0.1, (est, true)

    run = lambda data: subprocess.run([sys.executable, str(ROOT / 'scripts/calculate_avellaneda_parameters.py'), 'ETH'],
                                      env={**os.environ, 'HL_DATA_LOC': str(data), 'AVELLANEDA_PARAMS_DIR': str(tmp)},
                                      capture_output=True, text=True)
    r = run(Path(tmp, 'fresh'))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = json.loads(Path(tmp, 'avellaneda_parameters_ETH.json').read_text())
    assert isinstance(out['trade_enabled'], bool) and out['maker_fee'] == MAKER_FEE and out['data_end']
    write_synthetic(Path(tmp, 'stale'), now - 3600)
    assert run(Path(tmp, 'stale')).returncode != 0                     # collector dead for 1 h -> refuse


if __name__ == '__main__':
    check_effective_price_matches_reference()
    check_tick_size()
    check_backtest_quotes_equal_strategy_quotes()
    check_simulate_hand_cases()
    check_optimizer_known_limits()
    with tempfile.TemporaryDirectory() as tmp:
        check_retention_skips_old_files(tmp)
        check_book_is_causal(tmp)
        check_known_truth_and_calculator(tmp)
    print(f'OK (pandas {pd.__version__})')
