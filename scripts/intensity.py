"""Size-aware, interval-censored window maxima; one observation per quote exposure."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import mid_grid


def fit_maxima(depths, tick, seconds=15):
    """F(d)=exp(-A*seconds*exp(-k*d)); depths below one tick are left-censored."""
    depths = np.asarray(depths, float)
    bins = np.floor(np.maximum(depths, 0) / tick).astype(int)
    upper = bins + 1.0
    positive = bins > 0

    def nll(theta):
        a, kt = np.exp(theta)  # fit k in ticks to avoid price-unit conditioning
        log_fu = -a * seconds * np.exp(-kt * upper)
        log_fl = -a * seconds * np.exp(-kt * bins)
        terms = log_fu.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            terms[positive] += np.log(-np.expm1(log_fl[positive] - log_fu[positive]))
        return -float(terms.mean()) if np.isfinite(terms).all() else 1e100

    if not positive.any():
        raise ValueError("No identifiable crossing tail")
    k0 = 1 / max(float(np.mean(bins[positive])), 1)
    a0 = max(-np.log(max(np.mean(~positive), .01)) / seconds * np.exp(k0), 1e-5)
    fit = minimize(nll, np.log([a0, k0]), method="L-BFGS-B", bounds=[(-20, 8), (-12, 8)])
    if (not fit.success or not np.isfinite(fit.fun) or np.any(np.abs(fit.x - [-20, -12]) < 1e-4)
            or np.any(np.abs(fit.x - [8, 8]) < 1e-4)):
        raise ValueError("Intensity fit failed or is unidentifiable")
    a, kt = np.exp(fit.x)
    return float(a), float(kt / tick)


def window_depths(book, trades, start, end, quantity, seconds=15):
    """Full-quantity strict-crossing maxima; exclude incomplete or clock-uncertain windows."""
    if quantity <= 0:
        raise ValueError("Positive order quantity required")
    start, end = pd.Timestamp(start).ceil(f"{seconds}s"), pd.Timestamp(end).floor(f"{seconds}s")
    times = pd.date_range(start, end, freq=f"{seconds}s", inclusive="left")
    mids = mid_grid(book, start, end, seconds=1)
    valid = mids.notna().resample(f"{seconds}s").agg(["sum", "size"])
    good = (valid["sum"] == seconds) & (valid["size"] == seconds)
    refs = mids.reindex(times)
    accepted = times[good.reindex(times, fill_value=False).to_numpy() & refs.notna().to_numpy()]
    bad = trades[~trades.clock_valid]
    if not bad.empty:
        # A rejected print is unknown activity, not a zero-crossing observation.
        uncertain = bad.index.floor(f"{seconds}s").union(pd.DatetimeIndex(bad.received_at).floor(f"{seconds}s"))
        accepted = accepted[~accepted.isin(uncertain)]
    output = pd.DataFrame({"mid": refs.reindex(accepted), "bid_depth": 0., "ask_depth": 0.}, index=accepted)
    in_window = trades[(trades.index >= start) & (trades.index < end) & trades.clock_valid]
    for (at, side), group in in_window.groupby([in_window.index.floor(f"{seconds}s"), "side"]):
        if at not in output.index:
            continue
        sign = -1 if side == "sell" else 1
        depth = sign * (group.price.to_numpy() - output.at[at, "mid"])
        order = np.argsort(-depth)
        volume = group["size"].to_numpy()[order].cumsum()
        crossed = np.flatnonzero(volume >= quantity)
        if len(crossed):
            output.at[at, "bid_depth" if side == "sell" else "ask_depth"] = max(0., float(depth[order[crossed[0]]]))
    output.attrs["coverage"] = len(accepted) / max(len(times), 1)
    output.attrs["quantity"] = quantity
    return output


def estimate_intensity(depths, tick, seconds=15, min_windows=1000, min_events=30, bootstrap=32):
    """Fit window maxima; optional circular bootstrap groups successive accepted windows."""
    depths = np.asarray(depths, float)
    if not np.isfinite(depths).all() or tick <= 0:
        raise ValueError("Invalid crossing observations")
    events = int((depths >= tick).sum())
    if len(depths) < min_windows or events < min_events:
        raise ValueError(f"Insufficient intensity observations: {len(depths)} windows, {events} crossings")
    a, k = fit_maxima(depths, tick, seconds)
    # Quotes beyond this distance would be unsupported extrapolation.
    ordered = np.sort(depths)[::-1]
    max_delta = float(np.floor(np.nextafter(ordered[19], 0) / tick) * tick)
    if max_delta < tick:
        raise ValueError("Fewer than 20 supporting crossings beyond one tick")
    delta = np.linspace(tick, max_delta, min(20, max(2, int(max_delta / tick))))
    empirical = np.array([(depths > d).mean() for d in delta])
    predicted = -np.expm1(-a * seconds * np.exp(-k * delta))
    samples = []
    rng = np.random.default_rng(1729)
    block = max(1, 1800 // seconds)
    for _ in range(bootstrap):
        starts = rng.integers(0, len(depths), int(np.ceil(len(depths) / block)))
        indices = (starts[:, None] + np.arange(block)) % len(depths)
        try:
            samples.append(fit_maxima(depths[indices.ravel()[:len(depths)]], tick, seconds))
        except ValueError:
            pass
    intervals = np.quantile(samples, [.025, .975], axis=0).T.tolist() if len(samples) >= 10 else None
    return {"A": a, "k": k, "min_delta": tick, "max_delta": max_delta,
            "quantity": None, "windows": int(len(depths)), "crossings": events,
            "intervals_A_k": intervals, "bootstrap_successes": len(samples),
            "calibration": {"delta": delta.tolist(), "observed": empirical.tolist(),
                            "predicted": predicted.tolist()},
            "probability_rmse": float(np.sqrt(np.mean((predicted - empirical) ** 2)))}
