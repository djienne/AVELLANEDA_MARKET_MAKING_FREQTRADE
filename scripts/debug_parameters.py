import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add script dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_effective_mid_price, load_trades_data, get_tick_size
from volatility import calculate_volatility
from intensity import calculate_intensity_params
from backtest import optimize_gamma, run_backtest

def main():
    ticker = 'ETH'
    base_dir = "HL_data_collector/HL_data"
    orderbook_path = os.path.join(base_dir, f"orderbooks_{ticker}.parquet")
    trades_path = os.path.join(base_dir, f"trades_{ticker}.parquet")
    
    print(f"Loading data...")
    mid_price_df = load_effective_mid_price(orderbook_path)
    trades_df = load_trades_data(trades_path)
    buy_trades = trades_df[trades_df['side'] == 'buy'].copy()
    sell_trades = trades_df[trades_df['side'] == 'sell'].copy()
    
    print(f"Loaded {len(mid_price_df)} price points.")
    
    # Define periods (simulate what calculate_avellaneda_parameters does)
    N_minutes = 15
    H = N_minutes / 60.0
    
    max_time = mid_price_df.index.max()
    min_time = mid_price_df.index.min()
    
    valid_periods = []
    current_end = max_time
    
    # Just get the last 3 chunks
    for _ in range(3):
        current_start = current_end - pd.Timedelta(minutes=N_minutes)
        if current_start < min_time:
            break
        valid_periods.append(current_start)
        current_end = current_start
        
    list_of_periods = sorted(valid_periods)
    print(f"Analyzing {len(list_of_periods)} periods: {list_of_periods}")
    
    # 1. Volatility
    print("\n--- Volatility Debug ---")
    sigma_list = calculate_volatility(mid_price_df, H, list_of_periods)
    print(f"Sigmas: {sigma_list}")
    
    # 2. Intensity
    print("\n--- Intensity Debug ---")
    tick_size = get_tick_size(ticker)
    delta_list = np.arange(tick_size, 50.0 * tick_size, tick_size)
    
    A_bid_list, k_bid_list, A_ask_list, k_ask_list = calculate_intensity_params(
        list_of_periods, H, buy_trades, sell_trades, delta_list, mid_price_df
    )
    print(f"A_bid: {A_bid_list}")
    print(f"k_bid: {k_bid_list}")
    print(f"A_ask: {A_ask_list}")
    print(f"k_ask: {k_ask_list}")
    
    # 3. Backtest Debug
    print("\n--- Backtest Debug ---")
    ma_window = 3
    
    # Manually run the backtest logic for the last period
    j = len(list_of_periods) - 1
    if j >= 0:
        period_start = list_of_periods[j]
        period_end = period_start + pd.Timedelta(hours=H)
        print(f"Debugging backtest for period {j}: {period_start} to {period_end}")
        
        # Use params from previous period (or current if only 1)
        # In optimize_gamma, it uses previous periods.
        # Let's use the params we just calculated for this period to see if they work AT ALL
        sigma = sigma_list[j]
        k_bid = k_bid_list[j]
        k_ask = k_ask_list[j]
        
        print(f"Using params: sigma={sigma}, k_bid={k_bid}, k_ask={k_ask}")
        
        mask = (mid_price_df.index >= period_start) & (mid_price_df.index < period_end)
        s_df = mid_price_df.loc[mask]
        s = s_df.resample('s').asfreq(fill_value=np.nan).ffill()['mid_price']
        
        buy_mask = (buy_trades.index >= period_start) & (buy_trades.index < period_end)
        buy_trades_period = buy_trades.loc[buy_mask]
        sell_mask = (sell_trades.index >= period_start) & (sell_trades.index < period_end)
        sell_trades_period = sell_trades.loc[sell_mask]
        
        # Try a specific gamma
        gamma = 0.05
        print(f"Testing gamma={gamma}...")
        
        res = run_backtest(s, buy_trades_period, sell_trades_period, gamma, k_bid, k_ask, sigma, H)
        print(f"PnL: {res['pnl'][-1]}")
        print(f"Spread: {res['spread'][-1]}")
        print(f"Inventory: {res['q'][-1]}")
        print(f"Reservation: {res['r'][-1]}")
        print(f"Ask: {res['r_a'][-1]}")
        print(f"Bid: {res['r_b'][-1]}")
        
        if np.isnan(res['pnl'][-1]):
            print("PnL is NaN! Checking intermediate values...")
            print(f"Mid prices: {s.values[:5]} ...")
            print(f"Sigma abs: {sigma * s.values[-1]}")
            
            # Check for invalid values in arrays
            print(f"NaN in s: {np.isnan(s.values).any()}")
            print(f"NaN in r: {np.isnan(res['r']).any()}")
            print(f"NaN in spr: {np.isnan(res['spread']).any()}")

if __name__ == "__main__":
    main()
