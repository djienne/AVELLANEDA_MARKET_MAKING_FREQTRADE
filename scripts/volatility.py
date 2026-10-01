"""Causal log-return variance forecasts. GARCH must converge; EWMA is the explicit fallback."""
import warnings

import numpy as np
from arch import arch_model
from scipy.stats import chi2


def residual_diagnostics(residuals, lags=10):
    x = np.asarray(residuals, float)
    x = x[np.isfinite(x)]
    result = {}
    for label, values in (("residual", x), ("squared_residual", x ** 2)):
        values = values - values.mean()
        denom = float(values @ values)
        if len(values) <= lags or denom == 0:
            result[label + "_ljung_box_p"] = None
            continue
        acf = np.array([values[:-k] @ values[k:] / denom for k in range(1, lags + 1)])
        q = len(values) * (len(values) + 2) * np.sum(acf ** 2 / (len(values) - np.arange(1, lags + 1)))
        result[label + "_ljung_box_p"] = float(chi2.sf(q, lags))
    return result


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
    steps = horizon_seconds // sample_seconds
    ewma = returns.pow(2).ewm(halflife=3600 / sample_seconds, adjust=False, ignore_na=False).mean()
    variance = float(ewma.iloc[-1])
    forecasts = np.full(steps, variance)
    diagnostics = {"method": "ewma", "fallback_reason": None, "convergence_flag": None,
                   "coverage": float(mid.notna().mean()), "returns": int(len(valid)),
                   "ewma_variance_per_sample": variance}
    # A GARCH recursion must not compress a missing interval.
    gaps = np.flatnonzero(returns.isna().to_numpy())
    contiguous = returns.iloc[(gaps[-1] + 1 if len(gaps) else 0):]
    scale = float(contiguous.std()) if len(contiguous) > 1 else 0.0
    if len(contiguous) >= 500 and scale > 0 and np.isfinite(scale):
        try:
            with warnings.catch_warnings(record=True) as caught:
                result = arch_model(contiguous / scale, mean="Zero", vol="GARCH",
                                    p=1, q=1, dist="t", rescale=False).fit(
                                        disp="off", show_warning=False, options={"maxiter": 300})
            p = result.params
            diagnostics.update(convergence_flag=int(result.convergence_flag),
                               optimizer_message=str(result.optimization_result.message),
                               persistence=(float(p["alpha[1]"] + p["beta[1]"])
                                            if np.isfinite(p["alpha[1]"] + p["beta[1]"]) else None),
                               warning_count=len(caught))
            if not result.optimization_result.success or result.convergence_flag != 0:
                raise ValueError("GARCH optimizer did not converge")
            if not np.isfinite(p).all() or p["omega"] <= 0 or p["alpha[1]"] < 0 or p["beta[1]"] < 0:
                raise ValueError("Invalid GARCH parameters")
            if p["alpha[1]"] + p["beta[1]"] >= 1 or p["nu"] <= 2:
                raise ValueError("Nonstationary GARCH or infinite innovation variance")
            predicted = result.forecast(horizon=steps).variance.iloc[-1].to_numpy() * scale ** 2
            if not np.isfinite(predicted).all() or (predicted < 0).any():
                raise ValueError("Invalid forecast variance")
            forecasts = predicted
            diagnostics.update(method="garch", **residual_diagnostics(result.std_resid))
        except Exception as exc:
            diagnostics["fallback_reason"] = str(exc)
    else:
        diagnostics["fallback_reason"] = "Too few contiguous returns or constant prices"
    if not np.isfinite(forecasts).all() or (forecasts < 0).any():
        raise ValueError("Neither GARCH nor EWMA provided a valid forecast")
    return {"sample_seconds": sample_seconds,
            "cumulative_log_variance": np.r_[0.0, forecasts.cumsum()].tolist(),
            "sigma_daily": float(np.sqrt(forecasts[0] * 86400 / sample_seconds)),
            "diagnostics": diagnostics}
