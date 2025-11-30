# Avellaneda-Stoikov Market Making Model Parameter Calculator
# This script implements the optimal market making strategy from
# "High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008)

print("DEBUG: Script started", flush=True)
import numpy as np
print("DEBUG: numpy imported", flush=True)
import pandas as pd
print("DEBUG: pandas imported", flush=True)
import sys
import os
import argparse
from pathlib import Path
import json
print("DEBUG: Imports finished", flush=True)

# Import from modules
from utils import get_tick_size, load_trades_data, load_effective_mid_price
from volatility import calculate_volatility
from intensity import calculate_intensity_params
from backtest import optimize_gamma


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate Avellaneda-Stoikov market making parameters')
    parser.add_argument('ticker', nargs='?', default='ETH', help='Ticker symbol')
    parser.add_argument('--minutes', type=int, default=15, 
                        help='Frequency in minutes to recalculate parameters (default: 15)')
    return parser.parse_args()


def get_smoothed_parameters(param_list, ma_window, use_last_n=None):
    """
    Get smoothed parameter value using moving average.
    
    Args:
        param_list: List of parameter values
        ma_window: Number of periods to average
        use_last_n: If specified, only consider the last n values (default: use all)
    
    Returns:
        Smoothed parameter value, or NaN if insufficient data
    """
    if not param_list:
        return np.nan
    
    # FIX: Include the last element (was previously excluded due to off-by-one)
    if use_last_n is not None:
        values = param_list[-use_last_n:]
    else:
        values = param_list[-ma_window:] if ma_window > 0 else param_list[-1:]
    
    # Filter out NaN values
    valid_values = [v for v in values if pd.notna(v)]
    
    if not valid_values:
        return np.nan
    
    return np.mean(valid_values)


def calculate_final_quotes(gamma, sigma, A_bid, k_bid, A_ask, k_ask, H, mid_price_df, ma_window, ticker):
    """Calculate the final reservation price and quotes."""
    print("\n" + "-"*20)
    print("Calculating final parameters for current state...")
    
    s = mid_price_df['mid_price'].iloc[-1]
    time_remaining = H / 24.0
    q = 1.0  # Placeholder for current inventory

    # Convert percentage volatility to absolute volatility (in $)
    sigma_abs = sigma * s

    reservation_decay = gamma * sigma_abs**2.0 * time_remaining
    risk_aversion_term = 0.5 * reservation_decay
    
    half_spread_bid = risk_aversion_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_bid))
    half_spread_ask = risk_aversion_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_ask))
    spread_base = half_spread_bid + half_spread_ask
    
    r = s - q * reservation_decay
    gap = abs(r - s)

    # Calculate final quotes centered around reservation price
    r_a = r + half_spread_ask
    r_b = r - half_spread_bid
    
    # Calculate deltas relative to mid price
    delta_a = r_a - s
    delta_b = s - r_b
    
    return {
        "ticker": ticker,
        "timestamp": pd.Timestamp.now().isoformat(),
        "market_data": {
            "mid_price": float(s), 
            "sigma": float(sigma), 
            "A_bid": float(A_bid), "k_bid": float(k_bid),
            "A_ask": float(A_ask), "k_ask": float(k_ask)
        },
        "optimal_parameters": {"gamma": float(gamma)},
        "current_state": {
            "time_remaining": float(time_remaining), 
            "inventory": int(q), 
            "hours_window": H, 
            "ma_window": ma_window
        },
        "calculated_values": {
            "reservation_price": float(r), 
            "gap": float(gap), 
            "spread_base": float(spread_base), 
            "half_spread_bid": float(half_spread_bid),
            "half_spread_ask": float(half_spread_ask)
        },
        "limit_orders": {
            "ask_price": float(r_a), 
            "bid_price": float(r_b), 
            "delta_a": float(delta_a), 
            "delta_b": float(delta_b),
            "delta_a_percent": (delta_a / s) * 100.0, 
            "delta_b_percent": (delta_b / s) * 100.0
        }
    }


def print_summary(results, list_of_periods, script_dir):
    """Print a summary of the results to the terminal."""
    if not results:
        print("\n" + "="*80)
        print("AVELLANEDA-STOIKOV MARKET MAKING PARAMETERS")
        print("="*80)
        # FIX: Use ASCII-safe warning symbol
        print("[WARNING] DATA WARNING: Insufficient data for robust parameter estimation.")
        print("="*80)
        return

    TICKER = results['ticker']
    H = results['current_state']['hours_window']
    ma_window = results['current_state']['ma_window']
    
    print("\n" + "="*80)
    print(f"AVELLANEDA-STOIKOV MARKET MAKING PARAMETERS - {TICKER}")
    print(f"Analysis Period: {H * 60:.1f} minutes ({H:.4f} hours)")
    if ma_window > 1:
        print(f"Moving Average Window: {ma_window} periods")
    print("="*80)

    if len(list_of_periods) <= 1:
        print("[WARNING] DATA WARNING: Insufficient data for robust parameter estimation.")
        print("="*80)

    print(f"Market Data:")
    print(f"   Mid Price:                        ${results['market_data']['mid_price']:,.4f}")
    print(f"   Volatility (sigma):               {results['market_data']['sigma']:.6f}")
    print(f"   Intensity Bid (A_bid, k_bid):     A={results['market_data']['A_bid']:.4f}, k={results['market_data']['k_bid']:.6f}")
    print(f"   Intensity Ask (A_ask, k_ask):     A={results['market_data']['A_ask']:.4f}, k={results['market_data']['k_ask']:.6f}")
    print(f"\nOptimal Parameters:")
    print(f"   Risk Aversion (gamma): {results['optimal_parameters']['gamma']:.6f}")
    print(f"\nCurrent State:")
    print(f"   Time Remaining:        {results['current_state']['time_remaining']:.4f} (in days)")
    print(f"   Inventory (q):         {results['current_state']['inventory']:.4f}")
    print(f"\nCalculated Prices:")
    print(f"   Reservation Price:     ${results['calculated_values']['reservation_price']:.4f}")
    print(f"   Ask Price:             ${results['limit_orders']['ask_price']:.4f}")
    print(f"   Bid Price:             ${results['limit_orders']['bid_price']:.4f}")
    print(f"\nSpreads:")
    print(f"   Delta Ask:             ${results['limit_orders']['delta_a']:.6f} ({results['limit_orders']['delta_a_percent']:.6f}%)")
    print(f"   Delta Bid:             ${results['limit_orders']['delta_b']:.6f} ({results['limit_orders']['delta_b_percent']:.6f}%)")
    print(f"   Total Spread:          {(results['limit_orders']['delta_a_percent'] + results['limit_orders']['delta_b_percent']):.4f}%")
    
    json_filename = script_dir / f"avellaneda_parameters_{TICKER}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {json_filename}")
    print("="*80)


def main():
    """Main execution function."""
    args = parse_arguments()
    TICKER = args.ticker
    N_minutes = args.minutes
    H = N_minutes / 60.0
    
    # Determine MA window based on analysis period
    if H <= 8:
        ma_window = 3
    elif 8 < H < 20:
        ma_window = 2
    else:
        ma_window = 1

    print("-" * 20)
    print(f"DOING: {TICKER}")
    print(f"Using analysis period of {N_minutes} minutes ({H:.4f} hours).")
    if ma_window > 1:
        print(f"Using a {ma_window}-period moving average for parameters.")

    tick_size = get_tick_size(TICKER)
    delta_list = np.arange(tick_size, 50.0 * tick_size, tick_size)
    
    # Determine paths
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    project_root = script_dir.parent
    
    default_data_dir = project_root / 'HL_data_collector' / 'HL_data'
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    
    HL_DATA_DIR = os.getenv('HL_DATA_LOC', str(default_data_dir))
    print(f"Data directory: {HL_DATA_DIR}")
    
    # Load data
    parquet_file_path = os.path.join(HL_DATA_DIR, f'orderbooks_{TICKER}.parquet')

    if not os.path.exists(parquet_file_path):
        print(f"Error: Parquet file/directory {parquet_file_path} not found!")
        sys.exit(1)

    mid_price_df = load_effective_mid_price(parquet_file_path)
    trades_df = load_trades_data(os.path.join(HL_DATA_DIR, f'trades_{TICKER}.parquet'))
    buy_trades = trades_df[trades_df['side'] == 'buy'].copy()
    sell_trades = trades_df[trades_df['side'] == 'sell'].copy()
    print(f"Loaded {len(mid_price_df)} data points from {mid_price_df.index.min()} to {mid_price_df.index.max()}.")

    # Generate time chunks
    max_time = mid_price_df.index.max()
    min_time = mid_price_df.index.min()
    
    valid_periods = []
    max_chunks_limit = 50
    
    current_end = max_time
    chunks_found = 0
    
    while chunks_found < max_chunks_limit:
        current_start = current_end - pd.Timedelta(minutes=N_minutes)
        
        if current_start < min_time:
            # Check coverage for partial chunk
            overlap_end = current_end
            overlap_start = max(current_start, min_time)
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            target_seconds = N_minutes * 60.0
            
            if overlap_seconds / target_seconds >= 0.9:
                valid_periods.append(current_start)
                chunks_found += 1
            break
        else:
            valid_periods.append(current_start)
            chunks_found += 1
            current_end = current_start

    # Sort chronologically
    list_of_periods = sorted(valid_periods)

    print(f"Generated {len(list_of_periods)} chunks of {N_minutes} minutes.")

    if len(list_of_periods) < 3:
        print("Error: Fewer than 3 valid data chunks found. Need at least 3 chunks for parameter estimation.")
        sys.exit(1)

    # Calculate parameters
    sigma_list = calculate_volatility(mid_price_df, H, list_of_periods)
    A_bid_list, k_bid_list, A_ask_list, k_ask_list = calculate_intensity_params(
        list_of_periods, H, buy_trades, sell_trades, delta_list, mid_price_df
    )
    
    if len(list_of_periods) <= 1:
        print_summary({}, list_of_periods, script_dir)
        sys.exit()

    gammalist = optimize_gamma(
        list_of_periods, sigma_list, A_bid_list, k_bid_list, A_ask_list, k_ask_list,
        H, ma_window, mid_price_df, buy_trades, sell_trades, tick_size
    )

    # FIX: Consistent parameter selection for final calculation
    # Use the most recent values, applying MA smoothing if configured
    
    # Gamma
    if len(gammalist) > 0:
        gamma = get_smoothed_parameters(gammalist, ma_window)
        if pd.isna(gamma):
            # Fallback: try to get any valid value
            valid_gammas = [g for g in gammalist if pd.notna(g)]
            gamma = valid_gammas[-1] if valid_gammas else 0.1
    else:
        gamma = 0.1
    
    # Intensity parameters
    # FIX: Use the most recent values, not second-to-last
    A_bid = get_smoothed_parameters(A_bid_list, ma_window)
    k_bid = get_smoothed_parameters(k_bid_list, ma_window)
    A_ask = get_smoothed_parameters(A_ask_list, ma_window)
    k_ask = get_smoothed_parameters(k_ask_list, ma_window)
    
    # Fallback for intensity params if smoothing returned NaN
    DEFAULT_A = 1.0
    DEFAULT_K = 1.5
    
    if pd.isna(A_bid):
        valid_A = [a for a in A_bid_list if pd.notna(a)]
        A_bid = valid_A[-1] if valid_A else DEFAULT_A
        print(f"Warning: Using fallback A_bid={A_bid}")
    
    if pd.isna(k_bid):
        valid_k = [k for k in k_bid_list if pd.notna(k)]
        k_bid = valid_k[-1] if valid_k else DEFAULT_K
        print(f"Warning: Using fallback k_bid={k_bid}")
    
    if pd.isna(A_ask):
        valid_A = [a for a in A_ask_list if pd.notna(a)]
        A_ask = valid_A[-1] if valid_A else DEFAULT_A
        print(f"Warning: Using fallback A_ask={A_ask}")
    
    if pd.isna(k_ask):
        valid_k = [k for k in k_ask_list if pd.notna(k)]
        k_ask = valid_k[-1] if valid_k else DEFAULT_K
        print(f"Warning: Using fallback k_ask={k_ask}")
    
    # FIX: Sigma - use the most recent value, consistent with other params
    # The original code used [-2] which seems arbitrary
    sigma = get_smoothed_parameters(sigma_list, ma_window)
    if pd.isna(sigma):
        valid_sigmas = [s for s in sigma_list if pd.notna(s)]
        sigma = valid_sigmas[-1] if valid_sigmas else 0.01
        print(f"Warning: Using fallback sigma={sigma}")

    # Calculate and display results
    results = calculate_final_quotes(gamma, sigma, A_bid, k_bid, A_ask, k_ask, 
                                     H, mid_price_df, ma_window, TICKER)
    print_summary(results, list_of_periods, script_dir)


if __name__ == "__main__":
    main()
