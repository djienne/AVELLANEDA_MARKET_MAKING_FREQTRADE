"""Causal log-return variance forecast: one-hour-half-life EWMA of squared 5-s returns.

The forecast is flat. Student-t GARCH(1,1) on these returns fitted persistence ~1e-4 (no
usable clustering), and the risk term it feeds is small: at gamma=2, q=0.012 PAXG and
sigma_daily ~0.2 %, a full 30-min hold costs ~2e-4 USDC, ~3 % of one maker fee. Revisit a
term-structure model only if gamma*q^2*variance becomes comparable to fees.
"""
import numpy as np


def forecast_variance(mid, horizon_seconds=3600, sample_seconds=5):
    if horizon_seconds <= 0 or horizon_seconds % sample_seconds:
        raise ValueError("Horizon must be a positive multiple of sample interval")
    if len(mid) < 500 or mid.iloc[-1:].isna().any():
        raise ValueError("Insufficient or stale volatility observations")
    # Missing samples stay missing: a return must never bridge a gap.
    returns = np.log(mid.where(mid > 0)).diff()
    valid = returns.dropna()
    if len(valid) < 500:
        raise ValueError("Fewer than 500 observed adjacent returns")
    ewma = returns.pow(2).ewm(halflife=3600 / sample_seconds, adjust=False, ignore_na=False).mean()
    variance = float(ewma.iloc[-1])
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Invalid EWMA variance")
    forecasts = np.full(horizon_seconds // sample_seconds, variance)
    return {"sample_seconds": sample_seconds,
            "cumulative_log_variance": np.r_[0.0, forecasts.cumsum()].tolist(),
            "sigma_daily": float(np.sqrt(variance * 86400 / sample_seconds)),
            "diagnostics": {"method": "ewma", "coverage": float(mid.notna().mean()),
                            "returns": int(len(valid)), "ewma_variance_per_sample": variance}}
