
import numpy as np
import pandas as pd
import scipy.optimize

def calculate_intensity_params(list_of_periods, H, buy_orders, sell_orders, deltalist, mid_price_df):
    """
    Calculate order arrival intensity parameters (A and k) for bid and ask sides separately using MLE.
    
    The intensity model is: lambda(delta) = A * exp(-k * delta)
    where delta is the distance from mid-price to our quote.
    """
    print("\n" + "-"*20)
    print("Calculating order arrival intensity (A and k) separately for Bid and Ask...")

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

        # Ensure mid_price_df is sorted (it should be, but safety first)
        mid_price_subset = mid_price_df.loc[period_start:period_end].sort_index()
        
        if mid_price_subset.empty:
            A_bid_list.append(np.nan)
            k_bid_list.append(np.nan)
            A_ask_list.append(np.nan)
            k_ask_list.append(np.nan)
            continue

        # Data collection for fitting: (delta, count, duration)
        bid_data_points = []
        ask_data_points = []
        
        period_duration_seconds = H * 3600
        
        # --- BID SIDE ANALYSIS (Market Sells hitting our Bid) ---
        # We need to calculate how far each market sell "walked" down from the mid price.
        # Distance = Mid_Price - Trade_Price
        # A quote at 'delta' would be filled if Distance >= delta
        
        if not period_sell_orders.empty:
            # Align trades with mid prices
            # We need to reset index to use merge_asof
            sells_reset = period_sell_orders.reset_index().sort_values('datetime')
            mids_reset = mid_price_subset.reset_index().sort_values('datetime')
            
            merged_sells = pd.merge_asof(
                sells_reset, 
                mids_reset[['datetime', 'mid_price']], 
                on='datetime', 
                direction='backward'
            )
            
            # Calculate distance from mid
            # If trade price is 99 and mid is 100, distance is 1.
            merged_sells['distance'] = merged_sells['mid_price'] - merged_sells['price']
            
            # Filter out negative distances (trades above mid price - rare but possible in fast markets or crossed books)
            # For intensity estimation, we care about the "depth" side.
            valid_sells = merged_sells[merged_sells['distance'] > -1e-9]['distance'].values
        else:
            valid_sells = np.array([])

        # --- ASK SIDE ANALYSIS (Market Buys hitting our Ask) ---
        # We need to calculate how far each market buy "walked" up from the mid price.
        # Distance = Trade_Price - Mid_Price
        # A quote at 'delta' would be filled if Distance >= delta
        
        if not period_buy_orders.empty:
            buys_reset = period_buy_orders.reset_index().sort_values('datetime')
            mids_reset = mid_price_subset.reset_index().sort_values('datetime')
            
            merged_buys = pd.merge_asof(
                buys_reset, 
                mids_reset[['datetime', 'mid_price']], 
                on='datetime', 
                direction='backward'
            )
            
            merged_buys['distance'] = merged_buys['price'] - merged_buys['mid_price']
            valid_buys = merged_buys[merged_buys['distance'] > -1e-9]['distance'].values
        else:
            valid_buys = np.array([])

        # Calculate counts for each delta grid point
        for price_delta in deltalist:
            # Bid side: count sells that walked at least 'price_delta'
            n_bid_fills = np.sum(valid_sells >= price_delta) if len(valid_sells) > 0 else 0
            bid_data_points.append((price_delta, n_bid_fills, period_duration_seconds))

            # Ask side: count buys that walked at least 'price_delta'
            n_ask_fills = np.sum(valid_buys >= price_delta) if len(valid_buys) > 0 else 0
            ask_data_points.append((price_delta, n_ask_fills, period_duration_seconds))

        # Fit exponential model using MLE
        A_bid, k_bid = _fit_intensity_mle(bid_data_points)
        A_bid_list.append(A_bid)
        k_bid_list.append(k_bid)

        A_ask, k_ask = _fit_intensity_mle(ask_data_points)
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


def _fit_intensity_mle(data_points):
    """
    Fit the exponential intensity model to observed data points using MLE.
    Maximize Likelihood of Poisson process observations.
    
    Args:
        data_points: List of (delta, count, T) tuples.
        
    Returns:
        (A, k) parameters or (nan, nan).
    """
    # Unzip data
    deltas = np.array([p[0] for p in data_points])
    counts = np.array([p[1] for p in data_points])
    Ts = np.array([p[2] for p in data_points])
    
    # If no fills at all, cannot estimate A (it tends to 0) or k (undefined)
    if np.sum(counts) == 0:
        return np.nan, np.nan
        
    # Negative Log Likelihood function for Poisson Process
    # L = Product_j [ (lambda_j * T_j)^N_j * exp(-lambda_j * T_j) / N_j! ]
    # ln(L) = Sum_j [ N_j * ln(lambda_j * T_j) - lambda_j * T_j - ln(N_j!) ]
    # lambda_j = A * exp(-k * delta_j)
    # ln(lambda_j) = ln(A) - k * delta_j
    # Ignore constant terms (ln(N_j!), ln(T_j)) for optimization
    # NLL ~ Sum_j [ lambda_j * T_j - N_j * (ln(A) - k * delta_j) ]
    
    def nll(params):
        A, k = params
        if A <= 0 or k <= 0: return np.inf
        
        lambdas = A * np.exp(-k * deltas)
        term1 = lambdas * Ts
        # Use simple log algebra: N * ln(lambda) = N * (ln(A) - k*delta)
        term2 = counts * (np.log(A) - k * deltas)
        
        return np.sum(term1 - term2)

    # Initial guess heuristic
    # Use non-zero counts to estimate roughly via log-linear regression
    mask = counts > 0
    p0 = [1.0, 1.0] # Fallback
    
    if mask.sum() >= 2:
        try:
            # Empirical rate = count / T
            # ln(rate) = ln(A) - k * delta
            y = np.log(counts[mask] / Ts[mask])
            x = deltas[mask]
            slope, intercept = np.polyfit(x, y, 1)
            # Ensure guesses are positive
            k_guess = max(-slope, 0.01)
            A_guess = max(np.exp(intercept), 0.01)
            p0 = [A_guess, k_guess]
        except:
            pass
    elif mask.sum() == 1:
        # One point: assume k=1 and solve for A
        # count/T = A * exp(-k * delta) -> A = (count/T) * exp(k * delta)
        idx = np.where(mask)[0][0]
        k_guess = 1.0
        A_guess = (counts[idx] / Ts[idx]) * np.exp(k_guess * deltas[idx])
        p0 = [A_guess, k_guess]

    try:
        # Run optimization
        # method L-BFGS-B handles bounds efficiently
        res = scipy.optimize.minimize(
            nll, 
            p0, 
            bounds=[(1e-10, None), (1e-10, None)], 
            method='L-BFGS-B'
        )
        
        if res.success:
            A, k = res.x
            # Final sanity checks
            if not np.isfinite(A) or not np.isfinite(k):
                return np.nan, np.nan
            return A, k
        else:
            return np.nan, np.nan
            
    except Exception:
        return np.nan, np.nan
