import pandas as pd
import os
import glob
import sys

def inspect_file(filepath):
    print(f"\nInspecting {os.path.basename(filepath)}...")
    try:
        df = pd.read_parquet(filepath)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        if 'timestamp' in df.columns:
            ts_min = df['timestamp'].min()
            ts_max = df['timestamp'].max()
            print(f"  Timestamp range: {ts_min} to {ts_max} (Duration: {ts_max - ts_min:.2f}s)")
            print(f"  Datetime range: {pd.to_datetime(ts_min, unit='s')} to {pd.to_datetime(ts_max, unit='s')}")
        
        if 'side' in df.columns:
            print(f"  Sides: {df['side'].unique()}")
            print(f"  Side counts:\n{df['side'].value_counts()}")
            
    except Exception as e:
        print(f"  ERROR reading file: {e}")

def main():
    base_dir = "HL_data_collector/HL_data/orderbooks_ETH.parquet"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    files = glob.glob(os.path.join(base_dir, "*.parquet"))
    files.sort()
    
    print(f"Found {len(files)} parquet files in {base_dir}")
    
    total_rows = 0
    all_dfs = []
    
    for f in files:
        inspect_file(f)
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
            total_rows += len(df)
        except:
            pass

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal combined rows: {len(full_df)}")
        print(f"Combined timestamp range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
        
        # Check for duplicates
        dupes = full_df.duplicated(subset=['timestamp', 'side']).sum()
        print(f"Duplicate (timestamp, side) rows: {dupes}")

if __name__ == "__main__":
    main()
