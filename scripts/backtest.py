"""
Choose gamma by simulating the deployed bot (user_data/strategies/avellaneda.py) on recorded data.

The bot is long-only and alternates: flat -> one resting bid, long one unit -> one resting ask. Freqtrade re-places
the order every loop (process_throttle_secs = 15 s) at calculate_optimal_spreads with inventory q = 0:
    half_spread = 0.5 * gamma * (sigma * s)^2 * T + ln(1 + gamma / k) / gamma + fee * s
Only gamma * T enters the risk term, so T is fixed to the analysis window H and gamma is the single knob.
"""
import numpy as np
import pandas as pd

MAKER_FEE = 0.0002  # the strategy reads this back from the params JSON ("maker_fee")
STEP = '15s'        # = process_throttle_secs in user_data/config.json: one quote per bot loop


def smooth(values, ma_window):
    """Mean of the last `ma_window` per-period estimates (NaN-skipping), carried forward over gaps."""
    return pd.Series(values, dtype=float).rolling(ma_window, min_periods=1).mean().ffill()


def half_spreads(gamma, s, sigma, k_bid, k_ask, T_days, fee=MAKER_FEE):
    """Distance from mid to our bid and to our ask; same as calculate_optimal_spreads in the strategy (q = 0)."""
    risk = 0.5 * gamma * (sigma * s) ** 2 * T_days
    return (risk + np.log1p(gamma / k_bid) / gamma + fee * s,
            risk + np.log1p(gamma / k_ask) / gamma + fee * s)


def simulate(s, bid, ask, sell_min, buy_max, fee=MAKER_FEE):
    """
    One unit, long-only, alternating. Flat: the bid fills if a taker sold strictly below it during the step.
    Long: the ask fills if a taker bought strictly above it. At most one fill per step; the opposite order is
    quoted from the next step (Freqtrade sees the fill on its next loop). NaN mid or quote -> no fill.
    Returns PnL marked at the last valid mid, position after each step, and the number of fills.
    """
    n = len(s)
    pnl, q = np.zeros(n), np.zeros(n)
    cash, pos, fills, mark = 0.0, 0, 0, np.nan
    for i in range(n):
        if not np.isnan(s[i]):
            mark = s[i]
            if pos == 0 and sell_min[i] < bid[i]:
                cash, pos, fills = cash - bid[i] * (1 + fee), 1, fills + 1
            elif pos == 1 and buy_max[i] > ask[i]:
                cash, pos, fills = cash + ask[i] * (1 - fee), 0, fills + 1
        q[i] = pos
        pnl[i] = cash + (pos * mark if pos else 0.0)
    return pnl, q, fills


def optimize_params(periods, H, sigma_list, k_bid_list, k_ask_list, ma_window, mid_df, buy_trades, sell_trades,
                    fee=MAKER_FEE):
    """
    Simulate every gamma over all periods, each period quoted with sigma and k estimated on earlier periods
    (no look-ahead). Returns (gamma, trade_enabled, diagnostics).

    Edge = drift-adjusted PnL, dPnL - mean(position) * dMid: it removes what a long-only bot earns or loses from
    the price trend (luck) and keeps spread capture minus adverse selection. trade_enabled needs edge > 0 for the
    chosen gamma on the full window AND, out of sample, for the gamma chosen on the first 2/3 of the window.
    """
    T_days = H / 24.0
    start, end = periods[0], periods[-1] + pd.Timedelta(hours=H)
    grid = pd.date_range(start, end, freq=STEP, inclusive='left')
    s = mid_df['mid_price'].reindex(grid, method='ffill', tolerance=pd.Timedelta('60s')).to_numpy()

    def extreme(trades, how):  # most aggressive taker price in each step [t, t + STEP)
        px = trades['price'][(trades.index >= start) & (trades.index < end)]
        return px.resample(STEP, origin=start).agg(how).reindex(grid).to_numpy()
    buy_max, sell_min = extreme(buy_trades, 'max'), extreme(sell_trades, 'min')

    idx = pd.DatetimeIndex(periods).searchsorted(grid, side='right') - 1
    prev = lambda x: smooth(x, ma_window).shift(1).to_numpy()[idx]  # estimate from earlier periods only
    sig, kb, ka = prev(sigma_list), prev(k_bid_list), prev(k_ask_list)

    # Risk term 0.5*gamma*(sigma*s)^2*T spans 0.01/k .. 1000/k: from "inventory risk negligible" to "never filled"
    k_med, var_med = np.nanmedian(np.r_[kb, ka]), np.nanmedian((sig * s) ** 2)
    gammas = 2.0 * np.logspace(-2, 3, 41) / (k_med * var_med * T_days)
    if not np.all(np.isfinite(gammas)):
        print("No valid sigma/k estimates to simulate with; trading disabled.")
        return np.nan, False, {}

    n, m = len(grid), (2 * len(grid)) // 3
    mark = pd.Series(s).ffill().bfill().to_numpy()
    edge = lambda pnl, q, a, b: (pnl[b] - pnl[a]) - q[a:b].mean() * (mark[b] - mark[a])
    full, train, test, fills = [], [], [], []
    for g in gammas:
        hb, ha = half_spreads(g, s, sig, kb, ka, T_days, fee)
        pnl, q, nf = simulate(s, s - hb, s + ha, sell_min, buy_max, fee)
        full.append(edge(pnl, q, 0, n - 1)); train.append(edge(pnl, q, 0, m)); test.append(edge(pnl, q, m, n - 1))
        fills.append(nf)

    best, g_train = int(np.argmax(full)), int(np.argmax(train))
    trade_enabled = bool(full[best] > 0 and test[g_train] > 0)

    s_med = np.nanmedian(s)
    hb_med, _ = half_spreads(gammas, s_med, np.nanmedian(sig), np.nanmedian(kb), np.nanmedian(ka), T_days, fee)
    print(f"\nSimulated {n} steps of {STEP} (long-only, alternating). Edge in $ per unit, drift-adjusted:")
    print(f"{'gamma':>11} {'bid bp':>7} {'fills':>6} {'edge full':>10} {'edge 1/3 out':>13}")
    for i, g in enumerate(gammas):
        flag = " <- chosen" if i == best else (" <- train" if i == g_train else "")
        print(f"{g:11.4g} {hb_med[i] / s_med * 1e4:7.2f} {fills[i]:6d} {full[i]:10.4f} {test[i]:13.4f}{flag}")
    print(f"trade_enabled = {trade_enabled}  (full edge {full[best]:.4f}, out-of-sample edge of train gamma "
          f"{test[g_train]:.4f})")

    diagnostics = {
        "step": STEP, "steps": int(n), "maker_fee": fee,
        "gamma_grid": [float(g) for g in gammas],
        "fills": [int(f) for f in fills],
        "edge_full": [float(e) for e in full],
        "edge_holdout": [float(e) for e in test],
        "gamma_train": float(gammas[g_train]),
        "holdout_edge_of_train_gamma": float(test[g_train]),
    }
    return float(gammas[best]), trade_enabled, diagnostics
