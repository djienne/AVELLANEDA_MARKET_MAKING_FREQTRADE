import pandas as pd
import numpy as np
import os
import glob
import sys

def check_consistency(filepath, gap_threshold=5.0):
    print(f"\nChecking {filepath}...")
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"  ERROR: Could not read file: {e}")
        return

    if df.empty:
        print("  WARNING: File is empty.")
        return

    # 1. Check Timestamps
    if 'timestamp' in df.columns:
        # Sort just in case, though it should be sorted
        if not df['timestamp'].is_monotonic_increasing:
            print("  FAIL: Timestamps are NOT monotonic increasing!")
            # Show where
            diffs = df['timestamp'].diff()
            bad_indices = diffs[diffs < 0].index
            print(f"    Found {len(bad_indices)} out-of-order timestamps. First at index {bad_indices[0] if len(bad_indices) > 0 else '?'}")
        else:
            print("  PASS: Timestamps are monotonic increasing.")

        # Check for gaps
        diffs = df['timestamp'].diff()
        gaps = diffs[diffs > gap_threshold]
        if not gaps.empty:
            print(f"  WARNING: Found {len(gaps)} time gaps > {gap_threshold}s.")
            for idx, gap in gaps.head(5).items():
                ts = df.loc[idx, 'timestamp']
                print(f"    Gap of {gap:.2f}s at {pd.to_datetime(ts, unit='s')} (row {idx})")
        else:
            print(f"  PASS: No time gaps > {gap_threshold}s.")
    else:
        print("  ERROR: 'timestamp' column missing.")

    # 2. Check Price Ordering
    # Bids should be decreasing: bid_0 > bid_1 > ...
    # Asks should be increasing: ask_0 < ask_1 < ...
    
    bid_violations = 0
    ask_violations = 0
    
    # We assume depth up to 20, but check actual columns
    for i in range(19):
        # Bids
        col_curr = f'bid_price_{i}'
        col_next = f'bid_price_{i+1}'
        
        if col_curr in df.columns and col_next in df.columns:
            # Check where current <= next (violation, strictly decreasing expected, or at least non-increasing?)
            # Usually strictly decreasing for distinct levels, but let's check for non-increasing first (>=) is good.
            # Wait, bid 0 is highest. So bid_0 > bid_1. Violation if bid_0 <= bid_1.
            # Actually, standard orderbook: bid[0] > bid[1]. 
            # If prices are equal, that's weird for levels, but maybe possible if size is split? usually levels are distinct prices.
            # Let's assume strictly decreasing.
            
            mask = (df[col_curr] <= df[col_next]) & df[col_curr].notna() & df[col_next].notna()
            vios = mask.sum()
            if vios > 0:
                bid_violations += vios
                print(f"    Bid violation {col_curr} vs {col_next}: {vios} rows (e.g. row {mask.idxmax()}: {df.loc[mask.idxmax(), col_curr]} <= {df.loc[mask.idxmax(), col_next]})")

        # Asks
        col_curr = f'ask_price_{i}'
        col_next = f'ask_price_{i+1}'
        
        if col_curr in df.columns and col_next in df.columns:
            # Asks should be increasing: ask_0 < ask_1. Violation if ask_0 >= ask_1.
            mask = (df[col_curr] >= df[col_next]) & df[col_curr].notna() & df[col_next].notna()
            vios = mask.sum()
            if vios > 0:
                ask_violations += vios
                print(f"    Ask violation {col_curr} vs {col_next}: {vios} rows (e.g. row {mask.idxmax()}: {df.loc[mask.idxmax(), col_curr]} >= {df.loc[mask.idxmax(), col_next]})")

    if bid_violations == 0:
        print("  PASS: Bid prices are strictly decreasing.")
    else:
        print(f"  FAIL: Found {bid_violations} total bid ordering violations.")

    if ask_violations == 0:
        print("  PASS: Ask prices are strictly increasing.")
    else:
        print(f"  FAIL: Found {ask_violations} total ask ordering violations.")
        
    # 3. Check Spread (Ask 0 > Bid 0)
    if 'ask_price_0' in df.columns and 'bid_price_0' in df.columns:
        crossed = (df['ask_price_0'] <= df['bid_price_0']) & df['ask_price_0'].notna() & df['bid_price_0'].notna()
        if crossed.any():
            print(f"  FAIL: Found {crossed.sum()} crossed markets (Ask 0 <= Bid 0).")
            print(f"    Example at row {crossed.idxmax()}: Ask {df.loc[crossed.idxmax(), 'ask_price_0']} <= Bid {df.loc[crossed.idxmax(), 'bid_price_0']}")
        else:
            print("  PASS: No crossed markets (Ask 0 > Bid 0).")

def main():
    base_dir = "HL_data_collector/HL_data"
    pattern = os.path.join(base_dir, "orderbooks_*.parquet")
    dirs = glob.glob(pattern)
    
    if not dirs:
        print(f"No orderbook directories found matching {pattern}")
        return

    for d in dirs:
        print(f"\n{'='*80}")
        print(f"Checking directory: {d}")
        files = glob.glob(os.path.join(d, "*.parquet"))
        files.sort()
        
        for f in files:
            check_consistency(f)

if __name__ == "__main__":
    main()
