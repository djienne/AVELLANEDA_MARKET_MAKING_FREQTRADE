import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
import sys
import warnings
from numba import jit

# Minimum gamma value to avoid division by zero issues
MIN_GAMMA = 1e-8


def optimize_params(list_of_periods, sigma_list, A_bid_list, k_bid_list, A_ask_list, k_ask_list, 


                   H, ma_window, mid_price_df, buy_trades, sell_trades, tick_size):


    """


    Optimize risk aversion (gamma) and time horizon (T) via backtesting over ALL available data.


    


    Returns:


        (float, float): The optimal (gamma, time_horizon) tuple.


    """


    print("\n" + "-"*20)


    print("Optimizing parameters (gamma, time_horizon) via global backtesting...")





    # Determine start index (need at least one previous period for parameters)


    start_index = 1


    period_index_range = range(start_index, len(list_of_periods))





    # 1. Generate Grids


    


    # Representative parameters


    rep_s = mid_price_df['mid_price'].median()


    if pd.isna(rep_s):


        rep_s = mid_price_df['mid_price'].iloc[-1]


    


    valid_sigmas = [s for s in sigma_list if pd.notna(s)]


    rep_sigma = np.median(valid_sigmas) if valid_sigmas else 0.01


            


    all_k = [k for k in k_bid_list if pd.notna(k)] + [k for k in k_ask_list if pd.notna(k)]
    rep_k = np.median(all_k) if all_k else 1.0
            
    gamma_grid = generate_gamma_grid(rep_s, rep_sigma, rep_k, H)
    
    if gamma_grid is None or len(gamma_grid) == 0:
        print("Could not generate dynamic gamma grid. Using default grid.")
        gamma_grid = np.logspace(-3, 1, 32)  # 0.001 to 10
        
    # Time Horizon Grid: Explore multiples of the analysis window H
    # This represents the "urgency" to liquidate inventory
    # Generate 10 time horizons logarithmically spaced from 0.1*H to 10*H
    time_multipliers = np.geomspace(0.1, 10.0, 10)
    time_horizon_grid = [H * m for m in time_multipliers]

    print(f"Evaluating {len(gamma_grid)} gamma values x {len(time_horizon_grid)} time horizons across {len(period_index_range)} periods...")


    


    # Initialize PnL tracking for each (gamma, T) pair


    # Keys are tuples: (gamma, time_horizon)


    param_total_pnl = {}


    param_valid_periods = {}


    


    for g in gamma_grid:


        for t in time_horizon_grid:


            param_total_pnl[(g, t)] = 0.0


            param_valid_periods[(g, t)] = 0


    


    # 2. Iterate through all periods


    for j in period_index_range:


        # Get parameters from previous periods (avoid lookahead)


        if ma_window > 1:


            param_start = max(0, j - ma_window)


            param_end = j


            


            a_bid_slice = A_bid_list[param_start:param_end]


            k_bid_slice = k_bid_list[param_start:param_end]


            a_ask_slice = A_ask_list[param_start:param_end]


            k_ask_slice = k_ask_list[param_start:param_end]


            


            A_bid = pd.Series(a_bid_slice).dropna().mean()


            k_bid = pd.Series(k_bid_slice).dropna().mean()


            A_ask = pd.Series(a_ask_slice).dropna().mean()


            k_ask = pd.Series(k_ask_slice).dropna().mean()


        else:


            A_bid = A_bid_list[j - 1]


            k_bid = k_bid_list[j - 1]


            A_ask = A_ask_list[j - 1]


            k_ask = k_ask_list[j - 1]





        sigma = sigma_list[j - 1] if j > 0 else sigma_list[0]





        if pd.isna(sigma) or pd.isna(A_bid) or pd.isna(k_bid) or pd.isna(A_ask) or pd.isna(k_ask):


            continue


        


        if k_bid <= 0 or k_ask <= 0:


            continue





        period_start = list_of_periods[j]


        period_end = period_start + pd.Timedelta(hours=H)


        


        mask = (mid_price_df.index >= period_start) & (mid_price_df.index < period_end)


        s_df = mid_price_df.loc[mask]


        s = s_df.resample('s').asfreq(fill_value=np.nan).ffill()['mid_price']





        if s.empty or len(s) < 10:


            continue





        buy_mask = (buy_trades.index >= period_start) & (buy_trades.index < period_end)


        buy_trades_period = buy_trades.loc[buy_mask]


        sell_mask = (sell_trades.index >= period_start) & (sell_trades.index < period_end)


        sell_trades_period = sell_trades.loc[sell_mask]


        


        # Evaluate all parameter combinations for this period


        for gamma in gamma_grid:


            gamma_safe = max(gamma, MIN_GAMMA)


            


            for t_horizon in time_horizon_grid:


                # Evaluate


                res = evaluate_params(gamma_safe, t_horizon, s, buy_trades_period, sell_trades_period, 


                                        k_bid, k_ask, sigma)


                


                pnl = res[1]


                


                # Treat 0.0 PnL as valid (no trades case)


                if pd.notna(pnl):


                    param_total_pnl[(gamma, t_horizon)] += pnl


                    param_valid_periods[(gamma, t_horizon)] += 1


    


    print("\nBacktest complete.")


    


    # 3. Select Best Parameters


    best_gamma = 0.05


    best_time_horizon = H


    best_pnl = -float('inf')


    


    valid_results_found = False


    


    for g in gamma_grid:


        for t in time_horizon_grid:


            if param_valid_periods[(g, t)] > 0:


                pnl = param_total_pnl[(g, t)]


                valid_results_found = True


                


                # Maximizing PnL


                if pnl > best_pnl:


                    best_pnl = pnl


                    best_gamma = g


                    best_time_horizon = t


                # Tie-breaking: if PnL is identical (e.g. 0.0), prefer lower gamma (tighter spread)


                # to encourage trading, and lower time horizon (less inventory risk sensitivity)


                elif pnl == best_pnl:


                    if g < best_gamma:


                        best_gamma = g


                        best_time_horizon = t


    


    if not valid_results_found:


        print("Warning: No valid backtest results found. Using default parameters.")


        return 0.05, H


        


    print(f"Optimal Parameters: Gamma={best_gamma:.6f}, T={best_time_horizon:.4f}h (Total PnL: {best_pnl:.4f})")


    


    return best_gamma, best_time_horizon








def evaluate_params(gamma, time_horizon, mid_prices_period, buy_trades_period, sell_trades_period, 


                   k_bid, k_ask, sigma):


    """


    Run backtest for a single set of parameters.


    """


    if gamma <= MIN_GAMMA:


        return [gamma, np.nan]


    


    res = run_backtest(mid_prices_period, buy_trades_period, sell_trades_period, 


                       gamma, k_bid, k_ask, sigma, time_horizon)


    final_pnl = res['pnl'][-1]


    


    if not np.isfinite(final_pnl):


        return [gamma, np.nan]


        


    return [gamma, final_pnl]






def generate_gamma_grid(s, sigma, k, H):
    """
    Generate a grid of gamma values to test.
    
    Creates a logarithmically-spaced grid between gamma values that correspond
    to approximately 0.01% and 2% spreads.
    """
    time_remaining = (H / 2.0) / 24.0  # Half the horizon
    
    # Convert percentage volatility to absolute volatility
    sigma_abs = sigma * s
    
    # FIX: Guard against invalid parameters
    if sigma_abs <= 0 or k <= 0 or s <= 0:
        return None
    
    def spread(gamma):
        """Calculate spread as percentage of price."""
        if gamma <= MIN_GAMMA:
            return float('inf')
        try:
            term1 = gamma * sigma_abs**2 * time_remaining
            term2 = (2.0 / gamma) * np.log(1.0 + (gamma / k))
            return (term1 + term2) / s * 100.0
        except (OverflowError, ZeroDivisionError):
            return float('inf')
    
    try:
        gamma_001 = find_gamma(0.01, spread, k)
    except ValueError:
        _, gamma_001 = find_workable_spread(0.01, spread, k, 'up')
        if gamma_001 is None:
            return None
    
    try:
        gamma_1 = find_gamma(2.0, spread, k)
    except ValueError:
        _, gamma_1 = find_workable_spread(2.0, spread, k, 'down')
        if gamma_1 is None:
            return None
    
    # FIX: Ensure valid range
    gamma_min = max(MIN_GAMMA, min(gamma_1, gamma_001) * 0.99)
    gamma_max = max(gamma_1, gamma_001) * 1.01
    
    if gamma_min >= gamma_max:
        gamma_min, gamma_max = MIN_GAMMA, 10.0
        
    return np.logspace(np.log10(gamma_min), np.log10(gamma_max), 32)


def find_gamma(target_spread, spread_func, k):
    """Find gamma for a given target spread using numerical root finding."""
    
    def equation(gamma):
        if gamma <= MIN_GAMMA:
            return float('inf')
        try:
            return spread_func(gamma) - target_spread
        except:
            return float('inf')
    
    def is_valid(gamma, tolerance=1e-6):
        if gamma <= MIN_GAMMA:
            return False
        try:
            return abs(spread_func(gamma) - target_spread) < tolerance
        except:
            return False

    # Try Brent's method first
    try:
        gamma_min, gamma_max = MIN_GAMMA * 10, 1000.0
        f_min = equation(gamma_min)
        f_max = equation(gamma_max)
        if np.isfinite(f_min) and np.isfinite(f_max) and f_min * f_max < 0:
            gamma = brentq(equation, gamma_min, gamma_max)
            if is_valid(gamma):
                return gamma
    except:
        pass

    # Fall back to fsolve with multiple initial guesses
    for guess in [1.0, k, 0.1, 10.0, k * 10, k * 0.1, 0.01]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fsolve(equation, guess, full_output=True)
                if result[2] == 1 and is_valid(result[0][0]):
                    return result[0][0]
        except:
            continue
    
    raise ValueError(f"Could not find gamma for target_spread = {target_spread}")


def find_workable_spread(initial_spread, spread_func, k, direction='up', factor=1.05, max_iterations=100):
    """Find a workable spread if the target is not achievable."""
    spread = initial_spread
    for i in range(max_iterations):
        try:
            gamma = find_gamma(spread, spread_func, k)
            return spread, gamma
        except ValueError:
            spread *= factor if direction == 'up' else (1 / factor)
    return None, None


@jit(nopython=True, cache=True)
def jit_backtest_loop(s_values, buy_max_values, sell_min_values, fee, 
                      reservation_decay, half_spread_bid, half_spread_ask, min_spread_pct):
    """
    Core JIT-compiled backtest loop.
    
    Simulates the Avellaneda-Stoikov market making strategy:
    - Maintains inventory q
    - Places bid/ask quotes around reservation price
    - Tracks PnL including fees
    """
    N = len(s_values)
    
    # Pre-allocate arrays
    q = np.zeros(N + 1)    # Inventory
    x = np.zeros(N + 1)    # Cash
    pnl = np.zeros(N + 1)  # Mark-to-market PnL
    spr = np.zeros(N + 1)  # Spread
    r = np.zeros(N + 1)    # Reservation price
    r_a = np.zeros(N + 1)  # Ask quote
    r_b = np.zeros(N + 1)  # Bid quote
    
    for i in range(N):
        # Calculate reservation price: r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        r[i] = s_values[i] - q[i] * reservation_decay[i]
        
        # Calculate quotes centered around reservation price
        raw_r_a = r[i] + half_spread_ask[i]
        raw_r_b = r[i] - half_spread_bid[i]
        
        # Enforce minimum spread constraint relative to mid price
        min_half_dist = s_values[i] * (min_spread_pct / 2.0)
        
        r_a[i] = max(raw_r_a, s_values[i] + min_half_dist)
        r_b[i] = min(raw_r_b, s_values[i] - min_half_dist)
        
        spr[i] = r_a[i] - r_b[i]
        
        # Check for order fills
        # Our ASK is filled if a taker BUY crosses our ask price
        sell = 0
        if not np.isnan(buy_max_values[i]) and buy_max_values[i] >= r_a[i]:
            sell = 1
        
        # Our BID is filled if a taker SELL crosses our bid price
        buy = 0
        if not np.isnan(sell_min_values[i]) and sell_min_values[i] <= r_b[i]:
            buy = 1
        
        # Update inventory: buy increases, sell decreases
        q[i + 1] = q[i] + buy - sell
        
        # Update cash including fees
        cash_from_sell = r_a[i] * (1.0 - fee) if sell else 0.0
        cash_for_buy = r_b[i] * (1.0 + fee) if buy else 0.0
        x[i + 1] = x[i] + cash_from_sell - cash_for_buy
        
        # Mark-to-market PnL
        pnl[i + 1] = x[i + 1] + q[i + 1] * s_values[i]

    return pnl, x, q, spr, r, r_a, r_b


def run_backtest(mid_prices, buy_trades, sell_trades, gamma, k_bid, k_ask, sigma, time_horizon, fee=0.00030, min_spread_pct=0.0004):
    """
    Simulate the Avellaneda-Stoikov market making strategy with optimized parameters.
    
    Args:
        mid_prices: Series of mid-prices indexed by time
        buy_trades: DataFrame of taker buy orders (they hit our ask)
        sell_trades: DataFrame of taker sell orders (they hit our bid)
        gamma: Risk aversion parameter
        k_bid: Order arrival intensity parameter for bid side
        k_ask: Order arrival intensity parameter for ask side
        sigma: Volatility (daily, as decimal)
        time_horizon: Time horizon in HOURS (optimized fixed value)
        fee: Trading fee (default 3 bps)
        min_spread_pct: Minimum total spread as a percentage (default 0.04%)
    
    Returns:
        Dictionary with PnL, cash, inventory, spread, and quote arrays
    """
    # FIX: Guard against invalid gamma
    if gamma <= MIN_GAMMA:
        gamma = MIN_GAMMA
    
    time_index = mid_prices.index
    
    # FIX: Improved trade resampling
    # Resample to fixed grid for simulation
    resample_freq = '5s'
    
    # For taker buys: get maximum price (highest price they paid)
    if not buy_trades.empty:
        buy_trades_clean = buy_trades.groupby(level=0)['price'].max()
        buy_max = buy_trades_clean.resample(resample_freq).max()
    else:
        buy_max = pd.Series(dtype=float)
    
    # For taker sells: get minimum price (lowest price they accepted)
    if not sell_trades.empty:
        sell_trades_clean = sell_trades.groupby(level=0)['price'].min()
        sell_min = sell_trades_clean.resample(resample_freq).min()
    else:
        sell_min = pd.Series(dtype=float)
    
    # Resample mid prices to same frequency
    mid_prices_resampled = mid_prices.resample(resample_freq).first()
    
    # Create aligned time index
    all_times = mid_prices_resampled.index
    buy_max = buy_max.reindex(all_times)
    sell_min = sell_min.reindex(all_times)
    mid_prices_aligned = mid_prices_resampled.reindex(all_times).ffill()
    
    # Drop NaN mid prices
    valid_mask = mid_prices_aligned.notna()
    mid_prices_aligned = mid_prices_aligned[valid_mask]
    buy_max = buy_max[valid_mask]
    sell_min = sell_min[valid_mask]
    
    N = len(mid_prices_aligned)
    if N == 0:
        return {'pnl': np.array([0]), 'x': np.array([0]), 'q': np.array([0]),
                'spread': np.array([0]), 'r': np.array([0]), 'r_a': np.array([0]), 'r_b': np.array([0])}
    
    # Convert percentage volatility to absolute volatility
    sigma_abs = sigma * mid_prices_aligned.values
    
    # Avellaneda-Stoikov formulas with FIXED TIME HORIZON
    # Instead of decaying time, we use a constant factor derived from the optimized time_horizon
    time_factor_days = time_horizon / 24.0
    
    # reservation_decay is now constant w.r.t time (though varies with price via sigma_abs)
    reservation_decay = gamma * sigma_abs**2.0 * time_factor_days
    risk_aversion_term = 0.5 * reservation_decay
    
    # Half-spreads (from reservation price to quotes)
    half_spread_bid = risk_aversion_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_bid))
    half_spread_ask = risk_aversion_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_ask))
    
    # Run JIT-compiled simulation
    pnl, x, q, spr, r, r_a, r_b = jit_backtest_loop(
        mid_prices_aligned.values, buy_max.values, sell_min.values, fee,
        reservation_decay, half_spread_bid, half_spread_ask,
        min_spread_pct
    )
    
    return {'pnl': pnl, 'x': x, 'q': q, 'spread': spr, 'r': r, 'r_a': r_a, 'r_b': r_b}
