import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
import sys
import warnings
from numba import jit

# Minimum gamma value to avoid division by zero issues
MIN_GAMMA = 1e-8


def optimize_gamma(list_of_periods, sigma_list, A_bid_list, k_bid_list, A_ask_list, k_ask_list, 
                   H, ma_window, mid_price_df, buy_trades, sell_trades, tick_size):
    """
    Optimize risk aversion parameter (gamma) via backtesting.
    
    Uses historical periods to find the gamma that maximizes PnL while maintaining
    reasonable spreads.
    """
    print("\n" + "-"*20)
    print("Optimizing risk aversion (gamma) via backtesting...")

    GAMMA_CALCULATION_WINDOW = 4
    gammalist = []
    gamma_grid_to_test = None

    # FIX: Clarify index alignment
    # We use parameters from periods [0, 1, ..., j-1] to backtest period j
    # This avoids lookahead bias - we only use past information
    start_index = max(1, len(list_of_periods) - GAMMA_CALCULATION_WINDOW)
    period_index_range = range(start_index, len(list_of_periods))

    for j in period_index_range:
        # Get parameters from previous periods (avoid lookahead)
        if ma_window > 1:
            # Use moving average of previous periods' parameters
            param_start = max(0, j - ma_window)
            param_end = j  # Exclusive, so we use periods [param_start, j-1]
            
            a_bid_slice = A_bid_list[param_start:param_end]
            k_bid_slice = k_bid_list[param_start:param_end]
            a_ask_slice = A_ask_list[param_start:param_end]
            k_ask_slice = k_ask_list[param_start:param_end]
            
            # Filter out NaN values before averaging
            A_bid = pd.Series(a_bid_slice).dropna().mean()
            k_bid = pd.Series(k_bid_slice).dropna().mean()
            A_ask = pd.Series(a_ask_slice).dropna().mean()
            k_ask = pd.Series(k_ask_slice).dropna().mean()
        else:
            # Use parameters from immediately preceding period
            A_bid = A_bid_list[j - 1]
            k_bid = k_bid_list[j - 1]
            A_ask = A_ask_list[j - 1]
            k_ask = k_ask_list[j - 1]

        # Use sigma from previous period to avoid lookahead
        sigma = sigma_list[j - 1] if j > 0 else sigma_list[0]

        # Validate all parameters
        if pd.isna(sigma) or pd.isna(A_bid) or pd.isna(k_bid) or pd.isna(A_ask) or pd.isna(k_ask):
            print(f"Period {j}: Missing parameters, skipping.")
            gammalist.append(np.nan)
            continue
        
        # FIX: Ensure k values are positive to avoid log(1 + gamma/k) issues
        if k_bid <= 0 or k_ask <= 0:
            print(f"Period {j}: Invalid k values (k_bid={k_bid}, k_ask={k_ask}), skipping.")
            gammalist.append(np.nan)
            continue

        period_start = list_of_periods[j]
        period_end = period_start + pd.Timedelta(hours=H)
        print(f"\nProcessing period {j}: {period_start} to {period_end}")

        # Get mid-price data for this period
        mask = (mid_price_df.index >= period_start) & (mid_price_df.index < period_end)
        s_df = mid_price_df.loc[mask]
        s = s_df.resample('s').asfreq(fill_value=np.nan).ffill()['mid_price']

        if s.empty or len(s) < 10:
            print(f"Period {j}: Insufficient price data, skipping.")
            gammalist.append(np.nan)
            continue

        # Generate gamma grid if not already done
        if gamma_grid_to_test is None:
            k_avg = (k_bid + k_ask) / 2.0
            gamma_grid_to_test = generate_gamma_grid(s.iloc[-1], sigma, k_avg, H)

        if gamma_grid_to_test is None or len(gamma_grid_to_test) == 0:
            print("Could not find a reasonable gamma interval. Using default grid.")
            gamma_grid_to_test = np.logspace(-3, 1, 32)  # 0.001 to 10

        # Get trade data for this period
        buy_mask = (buy_trades.index >= period_start) & (buy_trades.index < period_end)
        buy_trades_period = buy_trades.loc[buy_mask]
        sell_mask = (sell_trades.index >= period_start) & (sell_trades.index < period_end)
        sell_trades_period = sell_trades.loc[sell_mask]

        # Test each gamma value
        gamma_results = []
        for gamma_to_test in gamma_grid_to_test:
            # FIX: Ensure gamma is above minimum threshold
            gamma_to_test = max(gamma_to_test, MIN_GAMMA)
            result = evaluate_gamma(gamma_to_test, s, buy_trades_period, sell_trades_period, 
                                    k_bid, k_ask, sigma, H)
            gamma_results.append(result)

        results_df = pd.DataFrame(gamma_results, columns=['gamma', 'pnl', 'spread'])
        valid_results = results_df.dropna(subset=['pnl'])
        
        if valid_results.empty:
            print("Warning: All backtests resulted in NaN PnL. Using fallback gamma.")
            best_gamma = 0.5
        else:
            # Prefer gamma with positive PnL and maximum spread (more conservative)
            positive_pnl_results = valid_results[valid_results['pnl'] > 0]
            if not positive_pnl_results.empty:
                best_gamma = positive_pnl_results.loc[positive_pnl_results['spread'].idxmax()]['gamma']
            else:
                # No profitable gamma found, use the one with least loss
                best_gamma = valid_results.loc[valid_results['pnl'].idxmax()]['gamma']
        
        print(f"Best gamma for period: {best_gamma:.5f}")
        gammalist.append(best_gamma)
        
    return gammalist


def evaluate_gamma(gamma, mid_prices_period, buy_trades_period, sell_trades_period, 
                   k_bid, k_ask, sigma, H):
    """
    Run backtest for a single gamma value and return results.
    
    Returns: [gamma, final_pnl, spread_base]
    """
    # FIX: Guard against invalid gamma
    if gamma <= MIN_GAMMA:
        return [round(gamma, 5), np.nan, np.nan]
    
    res = run_backtest(mid_prices_period, buy_trades_period, sell_trades_period, 
                       gamma, k_bid, k_ask, sigma, H)
    final_pnl = res['pnl'][-1]
    
    if not np.isfinite(final_pnl) or final_pnl == 0:
        return [round(gamma, 5), np.nan, np.nan]

    # Calculate spread for reporting
    s_mean = mid_prices_period.mean()
    sigma_abs_mean = sigma * s_mean
    
    # Calculate average spread (total bid + ask spread)
    risk_term = gamma * sigma_abs_mean**2.0 * 0.5
    half_spread_bid = risk_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_bid))
    half_spread_ask = risk_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_ask))
    spread_base = half_spread_bid + half_spread_ask
    
    return [round(gamma, 5), final_pnl, spread_base]


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
                      reservation_decay, half_spread_bid, half_spread_ask):
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
        r_a[i] = r[i] + half_spread_ask[i]
        r_b[i] = r[i] - half_spread_bid[i]
        
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


def run_backtest(mid_prices, buy_trades, sell_trades, gamma, k_bid, k_ask, sigma, H, fee=0.00030):
    """
    Simulate the Avellaneda-Stoikov market making strategy.
    
    Args:
        mid_prices: Series of mid-prices indexed by time
        buy_trades: DataFrame of taker buy orders (they hit our ask)
        sell_trades: DataFrame of taker sell orders (they hit our bid)
        gamma: Risk aversion parameter
        k_bid: Order arrival intensity parameter for bid side
        k_ask: Order arrival intensity parameter for ask side
        sigma: Volatility (daily, as decimal)
        H: Horizon in hours
        fee: Trading fee (default 3 bps)
    
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
    
    T = H / 24.0  # Horizon in days
    dt = T / N    # Time step
    
    # Prepare numpy arrays
    s_values = mid_prices_aligned.values
    buy_max_values = buy_max.values
    sell_min_values = sell_min.values
    
    # Time remaining at each step
    time_remaining = T - np.arange(N) * dt
    time_remaining = np.maximum(time_remaining, 0)  # Ensure non-negative
    
    # Convert percentage volatility to absolute volatility
    sigma_abs = sigma * s_values
    
    # Avellaneda-Stoikov formulas
    reservation_decay = gamma * sigma_abs**2.0 * time_remaining
    risk_aversion_term = 0.5 * reservation_decay
    
    # Half-spreads (from reservation price to quotes)
    half_spread_bid = risk_aversion_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_bid))
    half_spread_ask = risk_aversion_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_ask))
    
    # Run JIT-compiled simulation
    pnl, x, q, spr, r, r_a, r_b = jit_backtest_loop(
        s_values, buy_max_values, sell_min_values, fee,
        reservation_decay, half_spread_bid, half_spread_ask
    )
    
    return {'pnl': pnl, 'x': x, 'q': q, 'spread': spr, 'r': r, 'r_a': r_a, 'r_b': r_b}
