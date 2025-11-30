
import numpy as np
import pandas as pd
import scipy.optimize

def calculate_intensity_params(list_of_periods, H, buy_orders, sell_orders, deltalist, mid_price_df):
    """
    Calculate order arrival intensity parameters (A and k) for bid and ask sides separately.
    
    The intensity model is: lambda(delta) = A * exp(-k * delta)
    where delta is the distance from mid-price to our quote.
    
    - BID side intensity: How often do market SELL orders hit our bid quote?
      (Taker sells = our bid gets filled)
    - ASK side intensity: How often do market BUY orders hit our ask quote?
      (Taker buys = our ask gets filled)
    """
    print("\n" + "-"*20)
    print("Calculating order arrival intensity (A and k) separately for Bid and Ask...")

    def exp_fit(x, a, b):
        """Exponential decay function: lambda(delta) = a * exp(-b * delta)"""
        return a * np.exp(-b * x)

    A_bid_list, k_bid_list = [], []
    A_ask_list, k_ask_list = [], []

    for i in range(len(list_of_periods)):
        period_start = list_of_periods[i]
        period_end = period_start + pd.Timedelta(hours=H)

        # Get orders for this period
        mask_buy = (buy_orders.index >= period_start) & (buy_orders.index < period_end)
        period_buy_orders = buy_orders.loc[mask_buy].copy()

        mask_sell = (sell_orders.index >= period_start) & (sell_orders.index < period_end)
        period_sell_orders = sell_orders.loc[mask_sell].copy()

        if period_buy_orders.empty and period_sell_orders.empty:
            A_bid_list.append(np.nan)
            k_bid_list.append(np.nan)
            A_ask_list.append(np.nan)
            k_ask_list.append(np.nan)
            continue

        # Calculate reference mid-price for this period
        best_bid = period_buy_orders['price'].max() if not period_buy_orders.empty else np.nan
        best_ask = period_sell_orders['price'].min() if not period_sell_orders.empty else np.nan

        if pd.isna(best_bid) or pd.isna(best_ask):
            s_period = mid_price_df.loc[period_start:period_end]
            reference_mid = s_period['mid_price'].mean() if not s_period.empty else np.nan
        else:
            reference_mid = (best_bid + best_ask) / 2

        if pd.isna(reference_mid):
            A_bid_list.append(np.nan)
            k_bid_list.append(np.nan)
            A_ask_list.append(np.nan)
            k_ask_list.append(np.nan)
            continue

        # Calculate inter-arrival times for different delta levels
        interarrival_times_bid = {}
        interarrival_times_ask = {}
        
        period_duration_seconds = H * 3600
        
        for price_delta in deltalist:
            limit_bid_price = reference_mid - price_delta
            limit_ask_price = reference_mid + price_delta
            
            # BID side: Market SELL orders hitting our bid
            # A sell order hits our bid if sell_price <= our_bid_price
            bid_fill_times = []
            if not period_sell_orders.empty:
                sells_hitting_bid = period_sell_orders[period_sell_orders['price'] <= limit_bid_price]
                if not sells_hitting_bid.empty:
                    bid_fill_times = sells_hitting_bid.index.tolist()
            
            if len(bid_fill_times) > 1:
                hit_times = pd.DatetimeIndex(bid_fill_times)
                deltas = hit_times.to_series().diff().dt.total_seconds().dropna()
                # FIX: Filter out zero or negative inter-arrival times
                deltas = deltas[deltas > 0]
                if len(deltas) > 0:
                    interarrival_times_bid[price_delta] = deltas
                else:
                    interarrival_times_bid[price_delta] = pd.Series([period_duration_seconds])
            else:
                # No fills or only one fill - use period duration as proxy
                interarrival_times_bid[price_delta] = pd.Series([period_duration_seconds])

            # ASK side: Market BUY orders hitting our ask  
            # A buy order hits our ask if buy_price >= our_ask_price
            ask_fill_times = []
            if not period_buy_orders.empty:
                buys_hitting_ask = period_buy_orders[period_buy_orders['price'] >= limit_ask_price]
                if not buys_hitting_ask.empty:
                    ask_fill_times = buys_hitting_ask.index.tolist()
            
            if len(ask_fill_times) > 1:
                hit_times = pd.DatetimeIndex(ask_fill_times)
                deltas = hit_times.to_series().diff().dt.total_seconds().dropna()
                # FIX: Filter out zero or negative inter-arrival times
                deltas = deltas[deltas > 0]
                if len(deltas) > 0:
                    interarrival_times_ask[price_delta] = deltas
                else:
                    interarrival_times_ask[price_delta] = pd.Series([period_duration_seconds])
            else:
                interarrival_times_ask[price_delta] = pd.Series([period_duration_seconds])

        # Fit exponential model to BID side
        A_bid, k_bid = _fit_intensity_model(interarrival_times_bid, exp_fit, period_duration_seconds)
        A_bid_list.append(A_bid)
        k_bid_list.append(k_bid)

        # Fit exponential model to ASK side
        A_ask, k_ask = _fit_intensity_model(interarrival_times_ask, exp_fit, period_duration_seconds)
        A_ask_list.append(A_ask)
        k_ask_list.append(k_ask)

    # Print summary
    if A_bid_list and k_bid_list:
        print("Latest A and k values:")
        for i in range(max(0, len(A_bid_list) - 3), len(A_bid_list)):
            bid_str = f"A={A_bid_list[i]:.4f}, k={k_bid_list[i]:.6f}" if pd.notna(A_bid_list[i]) else "N/A"
            ask_str = f"A={A_ask_list[i]:.4f}, k={k_ask_list[i]:.6f}" if pd.notna(A_ask_list[i]) else "N/A"
            print(f"  - Bid: {bid_str} | Ask: {ask_str}")
    else:
        print("A and k values not available.")
        
    return A_bid_list, k_bid_list, A_ask_list, k_ask_list


def _fit_intensity_model(interarrival_times_dict, exp_fit, period_duration):
    """
    Fit the exponential intensity model to inter-arrival time data.
    
    Returns (A, k) parameters or (nan, nan) if fitting fails.
    """
    # Calculate lambda (arrival rate) for each delta level
    deltas = []
    lambdas = []
    
    for delta, times in interarrival_times_dict.items():
        mean_interarrival = times.mean()
        
        # FIX: Guard against division by zero
        if mean_interarrival > 0:
            lambda_val = 1.0 / mean_interarrival
        else:
            # If mean is zero, use a very small lambda (rare events)
            lambda_val = 1.0 / period_duration
        
        deltas.append(delta)
        lambdas.append(lambda_val)
    
    if len(deltas) < 2:
        return np.nan, np.nan
    
    deltas = np.array(deltas)
    lambdas = np.array(lambdas)
    
    # FIX: Filter out non-positive lambdas before fitting
    valid_mask = lambdas > 0
    if valid_mask.sum() < 2:
        return np.nan, np.nan
    
    deltas = deltas[valid_mask]
    lambdas = lambdas[valid_mask]
    
    try:
        # Initial guess: A = max(lambda), k = 1/median(delta)
        p0 = [lambdas.max(), 1.0 / np.median(deltas)]
        
        # Bounds to ensure positive parameters
        bounds = ([1e-10, 1e-10], [np.inf, np.inf])
        
        params, _ = scipy.optimize.curve_fit(
            exp_fit, 
            deltas, 
            lambdas, 
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )
        A, k = params
        
        # Sanity check: parameters should be reasonable
        if A <= 0 or k <= 0 or not np.isfinite(A) or not np.isfinite(k):
            return np.nan, np.nan
            
        return A, k
        
    except (RuntimeError, ValueError, scipy.optimize.OptimizeWarning) as e:
        return np.nan, np.nan
