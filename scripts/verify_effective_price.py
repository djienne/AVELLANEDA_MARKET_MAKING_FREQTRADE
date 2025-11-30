import pandas as pd
import numpy as np
import os
import sys
from utils import load_effective_mid_price, safe_read_parquet

def main():
    ticker = 'ETH'
    base_dir = "HL_data_collector/HL_data"
    parquet_path = os.path.join(base_dir, f"orderbooks_{ticker}.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return

    print(f"Loading data from {parquet_path}...")
    
    # Load raw df to get BBO
    raw_df = safe_read_parquet(parquet_path)
    if raw_df.empty:
        print("No data found.")
        return
        
    # Load effective df
    print("Calculating effective prices (threshold=$1000)...")
    effective_df = load_effective_mid_price(parquet_path, threshold=1000)
    
    print(f"\nComparing first 10 rows:")
    print(f"{'Timestamp':<20} | {'Bid (BBO)':<10} | {'Bid (Eff)':<10} | {'Diff':<10} | {'Ask (BBO)':<10} | {'Ask (Eff)':<10} | {'Diff':<10}")
    print("-" * 100)
    
    # Align timestamps (approximate since effective is resampled)
    # We'll just take the first few rows from effective_df and find matching raw rows
    
    count = 0
    for idx, row in effective_df.head(20).iterrows():
        ts = idx.timestamp()
        
        # Find raw row close to this timestamp
        # raw_df has 'timestamp' column
        mask = (raw_df['timestamp'] >= ts) & (raw_df['timestamp'] < ts + 1.0)
        matches = raw_df[mask]
        
        if not matches.empty:
            raw_row = matches.iloc[0]
            
            bbo_bid = raw_row['bid_price_0']
            eff_bid = row['price_bid']
            bid_diff = eff_bid - bbo_bid
            
            bbo_ask = raw_row['ask_price_0']
            eff_ask = row['price_ask']
            ask_diff = eff_ask - bbo_ask
            
            print(f"{idx} | {bbo_bid:<10.2f} | {eff_bid:<10.2f} | {bid_diff:<10.2f} | {bbo_ask:<10.2f} | {eff_ask:<10.2f} | {ask_diff:<10.2f}")
            count += 1
            if count >= 10:
                break

if __name__ == "__main__":
    main()
