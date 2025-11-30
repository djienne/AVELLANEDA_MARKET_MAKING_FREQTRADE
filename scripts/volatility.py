
import numpy as np
import pandas as pd
from arch import arch_model

# Use consistent return calculation method across all volatility estimators
# Log returns are preferred for financial applications (additive over time, symmetric)
USE_LOG_RETURNS = True


def calculate_garch_volatility(mid_price_df, H, list_of_periods):
    """
    Calculate volatility using GARCH(1,1) model with Student's t distribution.
    
    Returns a list of volatility estimates (one per period), expressed as 
    daily standard deviation of returns.
    """
    print("Calculating GARCH(1,1) Volatility...")
    
    if len(list_of_periods) < 2:
        print("Insufficient data for GARCH modeling.")
        return []
    
    sigma_garch_list = []
    
    for i, period_start in enumerate(list_of_periods):
        period_end = period_start + pd.Timedelta(hours=H)
        
        # Use all available data up to the end of the current period for estimation
        # This includes the current period and all prior history as requested
        mask = mid_price_df.index <= period_end
        historical_data = mid_price_df.loc[mask]
        
        if len(historical_data) < 100:
            sigma_garch_list.append(np.nan)
            continue

        if i == 0 or i == len(list_of_periods) - 1:
             print(f"  Period {i}: GARCH training data: {historical_data.index.min()} to {historical_data.index.max()} ({len(historical_data)} obs)")
        
        # FIX: Use consistent return calculation (log returns)
        if USE_LOG_RETURNS:
            returns = np.log(historical_data['mid_price']).diff().dropna()
        else:
            returns = historical_data['mid_price'].pct_change().dropna()
        
        if len(returns) < 100:
            sigma_garch_list.append(np.nan)
            continue
        
        # FIX: Calculate actual data frequency for proper scaling
        time_diffs = historical_data.index.to_series().diff().dropna()
        median_interval_seconds = time_diffs.dt.total_seconds().median()
        
        # Observations per day based on actual data frequency
        obs_per_day = 86400.0 / median_interval_seconds
        
        # Scale returns for numerical stability in GARCH estimation
        # GARCH works better with returns in the 0.1-10 range
        scale_factor = 100.0  # Convert to percentage
        returns_scaled = returns * scale_factor
        
        # Remove any infinite or NaN values
        returns_scaled = returns_scaled.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns_scaled) < 100:
            sigma_garch_list.append(np.nan)
            continue
        
        try:
            # Fit GARCH(1,1) model with Student's t distribution
            am = arch_model(
                returns_scaled, 
                mean='Constant', 
                vol='GARCH', 
                p=1, 
                q=1, 
                dist='t',
                rescale=False  # We've already scaled
            )
            res = am.fit(disp='off', show_warning=False)
            
            # Forecast conditional variance for the next period
            forecasts = res.forecast(horizon=1)
            variance_next = forecasts.variance.iloc[-1, 0]
            
            if variance_next <= 0 or not np.isfinite(variance_next):
                sigma_garch_list.append(np.nan)
                continue
            
            # Convert back from percentage to decimal
            volatility_per_obs = np.sqrt(variance_next) / scale_factor
            
            # FIX: Scale to daily volatility using actual observation frequency
            sigma_daily = volatility_per_obs * np.sqrt(obs_per_day)
            
            sigma_garch_list.append(sigma_daily)
            
            # Print model details for the latest period
            if i == len(list_of_periods) - 1:
                persistence = res.params.get('alpha[1]', 0) + res.params.get('beta[1]', 0)
                print(f"\nGARCH Model Results for latest period:")
                print(f"  Training Range: {historical_data.index.min()} to {historical_data.index.max()}")
                print(f"  Omega (α₀): {res.params.get('omega', np.nan):.6f}")
                print(f"  Alpha (α₁): {res.params.get('alpha[1]', np.nan):.6f}")
                print(f"  Beta (β₁):  {res.params.get('beta[1]', np.nan):.6f}")
                print(f"  Nu (df):    {res.params.get('nu', np.nan):.2f}")
                print(f"  Persistence: {persistence:.6f}")
                print(f"  Data frequency: {median_interval_seconds:.1f}s ({obs_per_day:.0f} obs/day)")
                
        except Exception as e:
            if i == len(list_of_periods) - 1:
                print(f"GARCH fitting failed: {e}")
            sigma_garch_list.append(np.nan)
    
    valid_sigmas = [s for s in sigma_garch_list if pd.notna(s)]
    
    if valid_sigmas:
        print(f"Calculated {len(valid_sigmas)} valid GARCH sigma values.")
        print("Latest GARCH volatility values:")
        for s in sigma_garch_list[-3:]:
            if pd.notna(s):
                print(f"  - {s:.6f}")
    else:
        print("No valid GARCH sigma values calculated.")
    
    return sigma_garch_list


def calculate_rolling_volatility(mid_price_df, H, list_of_periods):
    """
    Calculate rolling volatility (sigma) using a period-based window.
    
    Returns a list of volatility estimates (one per period), expressed as
    daily standard deviation of returns.
    """
    print("Calculating rolling volatility as fallback...")
    
    window_periods = 6  # Number of historical periods to use
    
    # Calculate volatility for each period
    period_volatilities = []
    
    for period_start in list_of_periods:
        period_end = period_start + pd.Timedelta(hours=H)
        mask = (mid_price_df.index >= period_start) & (mid_price_df.index < period_end)
        chunk = mid_price_df.loc[mask]
        
        if len(chunk) < 10:  # Need minimum data points
            period_volatilities.append(np.nan)
            continue
        
        # FIX: Use consistent return calculation (log returns, same as GARCH)
        if USE_LOG_RETURNS:
            returns = np.log(chunk['mid_price']).diff().dropna()
        else:
            returns = chunk['mid_price'].pct_change().dropna()
        
        if len(returns) < 2:
            period_volatilities.append(np.nan)
            continue
        
        # Calculate standard deviation of returns
        std_per_obs = returns.std()
        
        # FIX: Calculate actual data frequency for this chunk
        time_diffs = chunk.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_interval_seconds = time_diffs.dt.total_seconds().median()
            obs_per_day = 86400.0 / median_interval_seconds
        else:
            obs_per_day = 86400.0  # Assume 1-second data as fallback
        
        # Scale to daily volatility
        sigma_daily = std_per_obs * np.sqrt(obs_per_day)
        period_volatilities.append(sigma_daily)
    
    period_vol_series = pd.Series(period_volatilities, index=list_of_periods)
    
    num_periods = len(period_vol_series)
    
    # Adjust window if we have fewer periods than desired
    if num_periods < window_periods:
        actual_window = max(2, num_periods // 2) if num_periods > 1 else 1
    else:
        actual_window = window_periods
    
    print(f"Using rolling window of {actual_window} periods ({actual_window * H:.1f} hours)")
    
    # Apply exponentially weighted moving average for smoothing (more weight on recent)
    # Using span equivalent to window_periods
    smoothed_vol = period_vol_series.ewm(span=actual_window, min_periods=1).mean()
    
    sigma_list = smoothed_vol.tolist()
    
    if sigma_list:
        print("Latest rolling sigma values:")
        for s in sigma_list[-3:]:
            if pd.notna(s):
                print(f"  - {s:.6f}")
    else:
        print("Rolling sigma values not available.")
    
    return sigma_list


def calculate_volatility(mid_price_df, H, list_of_periods):
    """
    Calculate volatility (sigma) using GARCH and falling back to rolling 
    standard deviation for missing values.
    
    Returns a list of daily volatility estimates, one per period.
    """
    print("\n" + "-"*20)
    print("Calculating volatility (sigma)...")
    
    return_type = "log returns" if USE_LOG_RETURNS else "percentage returns"
    print(f"Using {return_type} for volatility calculation.")

    num_periods = len(list_of_periods)

    if num_periods < 3:
        print("Fewer than 3 periods available, using rolling volatility only.")
        final_sigma = calculate_rolling_volatility(mid_price_df, H, list_of_periods)
    else:
        # Calculate GARCH volatility
        garch_sigma = calculate_garch_volatility(mid_price_df, H, list_of_periods)

        # Calculate rolling volatility as a fallback
        rolling_sigma = calculate_rolling_volatility(mid_price_df, H, list_of_periods)

        # Ensure lists are of the same length
        while len(garch_sigma) < num_periods:
            garch_sigma.append(np.nan)
        while len(rolling_sigma) < num_periods:
            rolling_sigma.append(np.nan)

        garch_sigma = garch_sigma[:num_periods]
        rolling_sigma = rolling_sigma[:num_periods]

        # Combine results: prefer GARCH, fall back to rolling
        final_sigma = []
        garch_used = 0
        rolling_used = 0
        
        print("\nCombining GARCH and rolling volatility...")
        for i, (g, r) in enumerate(zip(garch_sigma, rolling_sigma)):
            use_garch = False
            if pd.notna(g) and g > 0:
                # Check for significant divergence if rolling is also available
                if pd.notna(r) and r > 0:
                    if g > 2 * r or g < r / 2:
                        # Divergence too high, prefer rolling
                        use_garch = False
                        if i >= num_periods - 3:
                            print(f"  - Period {i}: GARCH ({g:.6f}) differs significantly from Rolling ({r:.6f}). Using Rolling.")
                    else:
                        use_garch = True
                else:
                    # Only GARCH available
                    use_garch = True
            
            if use_garch:
                final_sigma.append(g)
                garch_used += 1
            elif pd.notna(r) and r > 0:
                final_sigma.append(r)
                rolling_used += 1
                if i >= num_periods - 3 and not (pd.notna(g) and g > 0): # Only log if it wasn't a divergence case (already logged)
                    print(f"  - Period {i}: GARCH failed/invalid, using rolling value: {r:.6f}")
            else:
                final_sigma.append(np.nan)
                if i >= num_periods - 3:
                    print(f"  - Period {i}: Both GARCH and rolling failed.")
        
        print(f"  Used GARCH for {garch_used}/{num_periods} periods, rolling for {rolling_used}/{num_periods}")
    
    # Forward fill any remaining NaNs, then backward fill if needed
    final_sigma_series = pd.Series(final_sigma)
    final_sigma_series = final_sigma_series.ffill().bfill()
    final_sigma = final_sigma_series.tolist()

    # Validate results
    if final_sigma and not all(pd.isna(s) for s in final_sigma):
        print("\nFinal combined sigma values:")
        for s in final_sigma[-3:]:
            if pd.notna(s):
                print(f"  - {s:.6f}")
            else:
                print("  - nan")
    else:
        print("\nCould not calculate any sigma values.")

    return final_sigma
