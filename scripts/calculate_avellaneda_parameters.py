# Avellaneda-Stoikov Market Making Model Parameter Calculator
# This script implements the optimal market making strategy from
# "High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008)

import numpy as np
import pandas as pd
import sys
import os
import time
import argparse
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = PROJECT_ROOT / "user_data" / "strategies"
if STRATEGY_DIR.exists():
    sys.path.append(str(STRATEGY_DIR))
try:
    from pair_loader import get_active_pair, pair_to_ticker
except Exception as exc:
    print(f"Warning: failed to import pair_loader ({exc}); defaulting to CLI ticker.", flush=True)
    get_active_pair = None
    pair_to_ticker = lambda pair: (pair or "").split("/")[0].split(":")[0].upper()

# Import from modules
from utils import get_tick_size, load_trades_data, load_effective_book, effective_mid_grid
from volatility import calculate_volatility
from intensity import calculate_intensity_params
from backtest import MAKER_FEE, half_spreads, optimize_params, smooth


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate Avellaneda-Stoikov market making parameters')
    parser.add_argument('ticker', nargs='?', default=None, help='Ticker symbol (defaults to first pair in config.json)')
    parser.add_argument('--minutes', type=int, default=15, 
                        help='Frequency in minutes to recalculate parameters (default: 15)')
    parser.add_argument('--max-data-age', type=float, default=600,
                        help='Refuse to run if the newest order book data is older than this many seconds '
                             '(default: 600; the collector rotates files every 300 s)')
    return parser.parse_args()


def resolve_ticker(cli_ticker: str | None) -> str:
    if cli_ticker:
        return cli_ticker.upper()
    if callable(get_active_pair) and callable(pair_to_ticker):
        try:
            pair = get_active_pair()
            ticker = pair_to_ticker(pair)
            if ticker:
                print(f"Using ticker from config pair: {pair} -> {ticker}", flush=True)
                return ticker
        except Exception as exc:
            print(f"Warning: failed to derive ticker from config: {exc}", flush=True)
    return "ETH"


def get_output_directory():
    """
    Get the output directory for parameter JSON files.
    Uses environment variable AVELLANEDA_PARAMS_DIR if set, otherwise defaults to scripts/ directory.
    Works consistently whether running locally, in Docker, or in a container.

    Returns:
        Path: Directory where parameter files should be written
    """
    # First check environment variable
    env_path = os.getenv('AVELLANEDA_PARAMS_DIR')
    if env_path:
        output_dir = Path(env_path).resolve()
        print(f"Using output directory from AVELLANEDA_PARAMS_DIR: {output_dir}")
        return output_dir

    # Fall back to scripts directory relative to project root
    # Try to find project root by looking for marker files
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    # If we're in scripts/ directory, use it directly
    if current_dir.name == 'scripts':
        output_dir = current_dir
    else:
        # Search for scripts directory
        search_paths = [
            current_dir / 'scripts',
            current_dir.parent / 'scripts',
            current_dir.parent.parent / 'scripts',
        ]

        for path in search_paths:
            if path.exists() and path.is_dir():
                output_dir = path
                break
        else:
            # If scripts/ not found, use current directory
            output_dir = current_dir
            print(f"Warning: scripts/ directory not found, using {output_dir}")

    print(f"Using default output directory: {output_dir}")
    return output_dir


def write_results(results, output_dir, tick_size):
    """Print a summary and write avellaneda_parameters_{TICKER}.json atomically."""
    md, op, fee = results['market_data'], results['optimal_parameters'], results['maker_fee']
    s = md['mid_price']
    hb, ha = half_spreads(op['gamma'], s, md['sigma'], md['k_bid'], md['k_ask'], op['time_horizon_hours'] / 24.0, fee)

    print("\n" + "=" * 80)
    print(f"AVELLANEDA-STOIKOV PARAMETERS - {results['ticker']}  (data up to {results['data_end']})")
    print("=" * 80)
    print(f"   sigma (daily): {md['sigma']:.6f}   k_bid: {md['k_bid']:.4f}/$   k_ask: {md['k_ask']:.4f}/$   tick: {tick_size:g}")
    print(f"   gamma: {op['gamma']:.6g}/$   T (fixed): {op['time_horizon_hours']:.4f} h   maker fee: {fee * 1e4:.1f} bp")
    print(f"   quotes around mid {s:,.4f}: bid -{hb / s * 1e4:.2f} bp, ask +{ha / s * 1e4:.2f} bp (fee included)")
    print(f"   periods used: {results['current_state']['num_data_periods']}   trade_enabled: {results['trade_enabled']}")
    if max(hb, ha) > 49 * tick_size:
        print(f"   Warning: quotes sit beyond the fitted delta range (49 ticks = {49 * tick_size:g} $); "
              f"k is extrapolated there.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"avellaneda_parameters_{results['ticker']}.json"
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(results, indent=4))
    os.replace(tmp, path)  # atomic: the bot never reads a half-written file
    print(f"Results saved to: {path}")
    print("=" * 80)


def main():
    """Main execution function."""
    args = parse_arguments()
    TICKER = resolve_ticker(args.ticker)
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

    output_dir = get_output_directory()
    default_data_dir = Path(__file__).resolve().parent.parent / 'HL_data_collector' / 'HL_data'
    HL_DATA_DIR = os.getenv('HL_DATA_LOC', str(default_data_dir))
    print(f"Data directory: {HL_DATA_DIR}")

    parquet_file_path = os.path.join(HL_DATA_DIR, f'orderbooks_{TICKER}.parquet')
    if not os.path.exists(parquet_file_path):
        print(f"Error: Parquet file/directory {parquet_file_path} not found!")
        sys.exit(1)

    book_df = load_effective_book(parquet_file_path)      # per snapshot: trade-vs-mid distances
    data_end = book_df.index.max()
    age_s = time.time() - data_end.timestamp()
    if age_s > args.max_data_age:
        print(f"Error: newest order book data is {age_s / 60:.1f} min old (limit {args.max_data_age / 60:.0f} min). "
              f"Is the collector running?")
        sys.exit(1)
    mid_price_df = effective_mid_grid(book_df)             # 1-s grid: volatility and backtest
    trades_df = load_trades_data(os.path.join(HL_DATA_DIR, f'trades_{TICKER}.parquet'))
    tick_size = get_tick_size(mid_price_df['mid_price'].iloc[-1])
    delta_list = np.arange(tick_size, 50.0 * tick_size, tick_size)
    print(f"Tick size (5 significant figures at last mid): {tick_size}")
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
        list_of_periods, H, buy_trades, sell_trades, delta_list, book_df
    )
    gamma, trade_enabled, backtest = optimize_params(
        list_of_periods, H, sigma_list, k_bid_list, k_ask_list, ma_window, mid_price_df, buy_trades, sell_trades
    )

    # Latest estimates, averaged over the last ma_window periods (same smoothing the simulation used)
    sigma, A_bid, k_bid, A_ask, k_ask = (smooth(x, ma_window).iloc[-1]
                                         for x in (sigma_list, A_bid_list, k_bid_list, A_ask_list, k_ask_list))
    if not np.all(np.isfinite([gamma, sigma, k_bid, k_ask])):
        print("Error: could not estimate sigma, k or gamma from the data; no parameters written.")
        sys.exit(1)

    results = {
        "ticker": TICKER,
        "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
        "data_end": data_end.tz_localize('UTC').isoformat(),
        "trade_enabled": trade_enabled,
        "maker_fee": MAKER_FEE,
        "market_data": {
            "mid_price": float(mid_price_df['mid_price'].iloc[-1]),
            "sigma": float(sigma),
            "A_bid": float(A_bid), "k_bid": float(k_bid),
            "A_ask": float(A_ask), "k_ask": float(k_ask),
            "tick_size": float(tick_size),
        },
        "optimal_parameters": {
            "gamma": float(gamma),
            "time_horizon_hours": float(H),
        },
        "current_state": {
            "analysis_window_hours": H,
            "ma_window": ma_window,
            "num_data_periods": len(list_of_periods),
        },
        "backtest": backtest,
    }
    write_results(results, output_dir, tick_size)


if __name__ == "__main__":
    main()
