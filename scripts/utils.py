
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def get_tick_size(ticker):
    """Get tick size based on the ticker symbol."""
    tick_sizes = {
        'BTC': 1.0,
        'ETH': 0.1,
        'SOL': 0.01,
        'WLFI': 0.0001,
        'PAXG': 0.01,
    }
    return tick_sizes.get(ticker, 0.01)


def safe_read_parquet(path):
    """
    Safely read parquet file or directory, skipping corrupted files.
    """
    path_obj = Path(path)
    
    if path_obj.is_file():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"Warning: Failed to read {path}. Skipping. Error: {e}")
            return pd.DataFrame()
            
    if path_obj.is_file():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            msg = str(e)
            if "Magic bytes not found" in msg:
                 print(f"Warning: Failed to read {path}. This might be an active file currently being written to. Skipping.")
            else:
                 print(f"Warning: Failed to read {path}. Skipping. Error: {e}")
            return pd.DataFrame()

            
    elif path_obj.is_dir():
        dfs = []
        files = list(path_obj.glob("*.parquet"))
        if not files:
            raise ValueError(f"No parquet files found in {path}")
             
        for p in files:
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                msg = str(e)
                if "Magic bytes not found" in msg:
                     # Silent skip for active files in directory scan to avoid spam
                     pass 
                else:
                     print(f"Warning: Skipping potentially incomplete/corrupted file: {p.name}. Error: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No valid data could be read from {path}")
        
        return pd.concat(dfs, ignore_index=True)

    else:
        raise ValueError(f"Path not found: {path}")


def load_trades_data(parquet_path):
    """Load trades data from a Parquet file/directory."""
    df = safe_read_parquet(parquet_path)
    
    if df.empty:
        raise ValueError(f"Parquet file at {parquet_path} is empty or all files were skipped.")
    
    if 'timestamp' not in df.columns:
        if 'timestamp' in df.index.names:
            df = df.reset_index()
        else:
            raise ValueError(f"Parquet file at {parquet_path} missing 'timestamp' column. Available columns: {df.columns.tolist()}")

    # Remove duplicates based on trade_id if available
    if 'trade_id' in df.columns:
        df = df.drop_duplicates(subset=['trade_id'])
    else:
        df = df.drop_duplicates()

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('datetime')
    df = df.sort_index()
    return df


def load_and_resample_mid_price(parquet_path):
    """Load and resample mid-price data from a Parquet file/directory."""
    df = safe_read_parquet(parquet_path)

    
    if df.empty:
        raise ValueError(f"Parquet file at {parquet_path} is empty or all files were skipped.")

    if 'timestamp' not in df.columns:
        if 'timestamp' in df.index.names:
            df = df.reset_index()
        else:
            raise ValueError(f"Parquet file at {parquet_path} missing 'timestamp' column. Available columns: {df.columns.tolist()}")

    # FIX: Add deduplication for prices data
    # Remove duplicate timestamps per side, keeping the last value
    df = df.drop_duplicates(subset=['timestamp', 'side'], keep='last')

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('datetime')
    
    # FIX: Validate that 'side' column contains expected values
    valid_sides = {'bid', 'ask'}
    actual_sides = set(df['side'].unique())
    if not actual_sides.intersection(valid_sides):
        raise ValueError(f"'side' column must contain 'bid' and/or 'ask'. Found: {actual_sides}")
    
    missing_sides = valid_sides - actual_sides
    if missing_sides:
        print(f"Warning: Missing sides in data: {missing_sides}")
    
    pivot_df = df.pivot_table(index='datetime', columns='side', values='price', aggfunc='last')
    
    # FIX: Check that required columns exist after pivot
    if 'bid' not in pivot_df.columns or 'ask' not in pivot_df.columns:
        raise ValueError(f"Pivot table missing required columns. Available: {pivot_df.columns.tolist()}")
    
    pivot_df.rename(columns={'bid': 'price_bid', 'ask': 'price_ask'}, inplace=True)
    
    # FIX: Only forward-fill once after resampling to avoid propagating stale data too far
    merged = pivot_df.resample('s').ffill()
    
    # FIX: Add a staleness check - don't forward fill beyond a threshold
    # Calculate time since last valid observation
    MAX_STALE_SECONDS = 60  # Don't trust data older than 60 seconds
    
    for col in ['price_bid', 'price_ask']:
        # Create a mask of original (non-ffilled) values
        # Resample the boolean notna() mask to seconds, taking max() (True if any update in that second)
        has_data_in_second = pivot_df[col].notna().resample('s').max().fillna(0).astype(bool)
        
        # Align with merged index
        original_mask = has_data_in_second.reindex(merged.index, fill_value=False)
        
        # Count consecutive NaNs (ffilled values)
        stale_count = (~original_mask).groupby((original_mask).cumsum()).cumcount()
        
        # Mark as NaN if too stale
        merged.loc[stale_count > MAX_STALE_SECONDS, col] = np.nan
    
    merged['mid_price'] = (merged['price_bid'] + merged['price_ask']) / 2
    merged.dropna(inplace=True)
    
    if merged.empty:

        raise ValueError("No valid mid-price data after processing. Check data quality.")
    
    return merged


def calculate_effective_prices(row, threshold=1000):
    """
    Calculate effective bid and ask prices based on cumulative volume threshold.
    """
    # Effective Bid
    effective_bid = np.nan
    cum_value = 0.0
    
    for i in range(20):
        price_col = f'bid_price_{i}'
        size_col = f'bid_size_{i}'
        
        if price_col not in row or size_col not in row:
            break
            
        price = row[price_col]
        size = row[size_col]
        
        if pd.isna(price) or pd.isna(size):
            break
            
        value = price * size
        cum_value += value
        
        if cum_value >= threshold:
            effective_bid = price
            break
            
    # Fallback to best bid if threshold not reached but some liquidity exists
    if pd.isna(effective_bid) and 'bid_price_0' in row and pd.notna(row['bid_price_0']):
        effective_bid = row['bid_price_0']

    # Effective Ask
    effective_ask = np.nan
    cum_value = 0.0
    
    for i in range(20):
        price_col = f'ask_price_{i}'
        size_col = f'ask_size_{i}'
        
        if price_col not in row or size_col not in row:
            break
            
        price = row[price_col]
        size = row[size_col]
        
        if pd.isna(price) or pd.isna(size):
            break
            
        value = price * size
        cum_value += value
        
        if cum_value >= threshold:
            effective_ask = price
            break
            
    # Fallback to best ask
    if pd.isna(effective_ask) and 'ask_price_0' in row and pd.notna(row['ask_price_0']):
        effective_ask = row['ask_price_0']
        
    return pd.Series({'price_bid': effective_bid, 'price_ask': effective_ask})


def load_effective_mid_price(parquet_path, threshold=1000):
    """
    Load orderbook data and calculate effective mid-price based on depth.
    """
    df = safe_read_parquet(parquet_path)
    
    if df.empty:
        raise ValueError(f"Parquet file at {parquet_path} is empty or all files were skipped.")

    if 'timestamp' not in df.columns:
        if 'timestamp' in df.index.names:
            df = df.reset_index()
        else:
            raise ValueError(f"Parquet file at {parquet_path} missing 'timestamp' column.")

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('datetime')
    df = df.set_index('datetime')
    
    # Calculate effective prices
    # This might be slow for very large datasets, but acceptable for 15 min chunks
    effective_prices = df.apply(lambda row: calculate_effective_prices(row, threshold), axis=1)
    
    # Resample to 1s
    merged = effective_prices.resample('s').last().ffill()
    
    # Staleness check
    MAX_STALE_SECONDS = 60
    
    for col in ['price_bid', 'price_ask']:
        # Check original data presence (resampled to seconds)
        has_data = effective_prices[col].notna().resample('s').max().fillna(0).astype(bool)
        original_mask = has_data.reindex(merged.index, fill_value=False)
        stale_count = (~original_mask).groupby((original_mask).cumsum()).cumcount()
        merged.loc[stale_count > MAX_STALE_SECONDS, col] = np.nan
        
    merged['mid_price'] = (merged['price_bid'] + merged['price_ask']) / 2
    merged.dropna(inplace=True)
    
    if merged.empty:
        raise ValueError("No valid effective mid-price data after processing.")
        
    return merged
