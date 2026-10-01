"""Constrained flat/long expected-wealth policy. Quantities are base units; wealth is USDC."""
import math

import numpy as np
import pandas as pd

from utils import get_tick_size, round_price, utc

STEP = 15
HORIZON = 1800
GAMMA_USDC = 2.0
STAKE = 50.0
POSITION_STOP = .01
TRIAL_DRAWDOWN = 10.0
TRIAL_DAYS = 7


def number(value, name, minimum=0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"Invalid {name}")
    return float(value)


def validate_params(p, pair, now):
    if not isinstance(p, dict) or p.get("pair") != pair:
        raise ValueError("Wrong parameter market")
    for key in ("data_valid", "trade_enabled"):
        if type(p.get(key)) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not p["data_valid"]:
        raise ValueError("; ".join(p.get("reasons", ["Data/model invalid"])))
    now = utc(now)
    start, end, calculated, expires = (utc(p[k]) for k in ("data_start", "data_end", "timestamp", "expires_at"))
    trades_end = utc(p["trade_data_end"])
    if not start < end <= calculated <= now + pd.Timedelta(seconds=1):
        raise ValueError("Invalid parameter chronology")
    if not start <= trades_end <= calculated:
        raise ValueError("Invalid trade chronology")
    if now > expires or expires > min(end, trades_end) + pd.Timedelta(minutes=30) or expires <= calculated:
        raise ValueError("Expired parameters")
    if (end - start).total_seconds() < 6 * 3600 or number(p["coverage"], "coverage") < .95:
        raise ValueError("Insufficient data coverage")
    market = p["market"]
    if market["symbol"] != pair or type(market["sz_decimals"]) is not int or not 0 <= market["sz_decimals"] <= 6:
        raise ValueError("Invalid market precision")
    for name in ("maker_fee", "taker_fee", "min_notional", "min_amount"):
        number(market[name], name)
    if max(market["maker_fee"], market["taker_fee"]) >= .01:
        raise ValueError("Invalid fee units")
    if not pd.Timedelta(0) <= now - utc(market["timestamp"]) <= pd.Timedelta(days=1):
        raise ValueError("Stale market metadata")
    number(p["gamma_usdc"], "gamma_usdc", strict=True)
    number(p["quantity"], "quantity", strict=True)
    number(p["reference_mid"], "reference_mid", strict=True)
    number(abs(p["funding_rate"]), "funding_rate")
    curve = np.asarray(p["volatility"]["cumulative_log_variance"], float)
    if p["volatility"]["sample_seconds"] != 5 or curve.ndim != 1 or len(curve) < 2 * HORIZON // 5 + 1:
        raise ValueError("Incomplete forecast horizon")
    if not np.isfinite(curve).all() or curve[0] != 0 or np.any(np.diff(curve) < 0):
        raise ValueError("Invalid cumulative variance")
    if not pd.Timedelta(0) <= end - utc(p["volatility"]["origin"]) <= pd.Timedelta(seconds=5):
        raise ValueError("Variance forecast origin differs from data cutoff")
    if expires > utc(p["volatility"]["origin"]) + pd.Timedelta(seconds=HORIZON):
        raise ValueError("Expiry exceeds the supported forecast")
    for side in ("bid", "ask"):
        model = p["intensity"][side]
        for key in ("A", "k", "min_delta", "max_delta"):
            number(model[key], f"{side}.{key}", strict=True)
        if model["max_delta"] < model["min_delta"] or model["windows"] < 1000 or model["crossings"] < 30:
            raise ValueError("Unsupported intensity fit")
        if model["quantity"] != p["quantity"]:
            raise ValueError("Intensity calibration quantity mismatch")
    if p["trade_enabled"] and p.get("evidence", {}).get("status") != "passed":
        raise ValueError("Evidence gate mismatch")
    return p


def validate_book(book, now, max_age=2):
    now_s = utc(now).timestamp()
    event = number(book.get("timestamp"), "book timestamp", strict=True) / 1000
    received = number(book.get("received_at", event), "receipt timestamp", strict=True)
    if not -1 <= now_s - event <= max_age or not -1 <= now_s - received <= max_age:
        raise ValueError("Stale/future order book")
    for side, sign in (("bids", -1), ("asks", 1)):
        levels = np.asarray(book.get(side, []), float)
        if levels.ndim != 2 or levels.shape[1] < 2 or not len(levels):
            raise ValueError("Empty order book")
        if not np.isfinite(levels[:, :2]).all() or np.any(levels[:, :2] <= 0):
            raise ValueError("Invalid order book")
        if np.any(sign * np.diff(levels[:, 0]) < 0):
            raise ValueError("Unsorted order book")
    if book["bids"][0][0] >= book["asks"][0][0]:
        raise ValueError("Crossed order book")
    return (book["bids"][0][0] + book["asks"][0][0]) / 2


def liquidation(book, quantity, fee):
    remaining, gross, limit = quantity, 0., book["bids"][0][0]
    for price, size, *_ in book["bids"]:
        fill = min(size, remaining)
        gross += fill * price
        remaining -= fill
        limit = price
        if remaining < 1e-12:
            return gross * (1 - fee), float(limit)
    raise ValueError("Insufficient visible liquidation depth")


def next_unlock(timestamp):
    return (math.floor(float(timestamp) / 900) + 1) * 900


def _quotes(mid, touch, model, sz_decimals, buy):
    tick = get_tick_size(mid, sz_decimals)
    lower = max(model["min_delta"], abs(mid - touch))
    upper = min(model["max_delta"], .019 * mid)
    n = int(math.floor((upper - lower) / tick)) + 1
    if n <= 0:
        return np.array([])
    # Cap wide tails at 512 candidates to bound runtime; wider fits need a grid-convergence check.
    offsets = np.unique(np.linspace(0, n - 1, min(n, 512), dtype=int))
    prices = np.array([round_price(mid + (-1 if buy else 1) * (lower + i * tick), sz_decimals, buy)
                       for i in offsets])
    distance = abs(prices - mid)
    good = (distance >= model["min_delta"] - 1e-10) & (distance <= upper + 1e-10) & (prices > 0)
    return np.unique(prices[good])


def _arrival(prices, mid, model):
    rate = model["A"] * np.exp(-model["k"] * abs(prices - mid))
    probability = -np.expm1(-rate * STEP)
    occupied = np.divide(probability, rate * STEP, out=np.ones_like(rate), where=rate > 1e-15)
    return probability, occupied


def quote_decision(p, book, now, quantity=0., remaining_seconds=HORIZON, stake=STAKE, planning_seconds=HORIZON):
    """Backward dynamic programming; actual partial quantity gets its own liquidation state.

    Reference mid and arrival coefficients are frozen over this receding horizon. The
    variance term evolves. Fill probabilities are a public-data proxy, not a queue model.
    """
    mid = validate_book(book, now)
    market = p["market"]
    decimals, fm, ft = market["sz_decimals"], market["maker_fee"], market["taker_fee"]
    unit = 10.0 ** -decimals
    target = min(math.floor(stake / book["bids"][0][0] / unit + 1e-9) * unit, p["quantity"])
    if target < market["min_amount"] or target * mid < market["min_notional"]:
        return {"action": "wait", "price": None, "quantity": 0., "value": 0., "reason": "minimum_size"}
    target = round(target, decimals)
    bid = _quotes(mid, book["bids"][0][0], p["intensity"]["bid"], decimals, True)
    ask = _quotes(mid, book["asks"][0][0], p["intensity"]["ask"], decimals, False)
    pb, ob = _arrival(bid, mid, p["intensity"]["bid"])
    pa, oa = _arrival(ask, mid, p["intensity"]["ask"])
    curve = np.asarray(p["volatility"]["cumulative_log_variance"], float)
    age = max(0., (utc(now) - utc(p["volatility"]["origin"])).total_seconds())
    at = np.arange(planning_seconds // STEP + 1) * STEP + age
    x = np.arange(len(curve)) * 5
    if at[-1] > x[-1] + 1e-6:
        raise ValueError("Planning horizon exceeds published variance forecast")
    cumulative = np.interp(at, x, curve)
    variance = np.diff(cumulative) * mid ** 2
    n = len(variance)
    flat = np.zeros(n + 1)
    quantities = [target] + ([quantity] if quantity > 0 else [])
    held = np.zeros((len(quantities), n + 1))
    costs, limits = [], []
    for j, q in enumerate(quantities):
        proceeds, limit = liquidation(book, q, ft)
        costs.append(q * mid - proceeds)
        limits.append(limit)
        held[j, -1] = -costs[j]
    chosen = None
    base = utc(now).timestamp()
    remaining_steps = max(0, int(math.ceil(remaining_seconds / STEP)))
    funding_rate = float(p.get("funding_rate", 0))
    for i in range(n - 1, -1, -1):
        unlock_now = min(n, max(i + 1, int(math.ceil((next_unlock(base + i * STEP) - base) / STEP))))
        unlock_fill = min(n, max(i + 1, int(math.ceil((next_unlock(base + (i + 1) * STEP) - base) / STEP))))
        funding_due = math.floor((base + (i + 1) * STEP) / 3600) > math.floor((base + i * STEP) / 3600)
        for j, q in enumerate(quantities):
            risk = .5 * p["gamma_usdc"] * q ** 2 * variance[i]
            funding = q * mid * funding_rate if funding_due else 0.
            candidates = [-costs[j] + flat[unlock_now], held[j, i + 1] - risk - funding]
            if len(ask):
                proceeds = q * (ask - mid) - fm * q * ask
                candidates.extend(pa * (proceeds + flat[unlock_fill]) + (1 - pa) * held[j, i + 1]
                                  - risk * oa - funding * (1 - pa))
            if quantity > 0 and j == len(quantities) - 1 and i >= remaining_steps:
                pick = 0
            else:
                pick = int(np.argmax(candidates))  # ties prefer liquidation, then waiting
            held[j, i] = candidates[pick]
            if i == 0 and quantity > 0 and j == len(quantities) - 1:
                chosen = {"action": "liquidate" if pick == 0 else ("wait" if pick == 1 else "sell"),
                          "price": limits[j] if pick == 0 else (None if pick == 1 else float(ask[pick - 2])),
                          "quantity": quantity, "value": float(candidates[pick]), "reason": "model"}
        scores = [flat[i + 1]]
        if len(bid):
            proceeds = target * (mid - bid) - fm * target * bid
            scores.extend(pb * (proceeds + held[0, i + 1]) + (1 - pb) * flat[i + 1]
                          - .5 * p["gamma_usdc"] * target ** 2 * variance[i] * (1 - ob)
                          - (target * mid * funding_rate * pb if funding_due else 0))
        pick = int(np.argmax(scores))
        flat[i] = scores[pick]
        if i == 0 and quantity <= 0:
            chosen = {"action": "wait" if pick == 0 else "buy",
                      "price": None if pick == 0 else float(bid[pick - 1]),
                      "quantity": target, "value": float(scores[pick]), "reason": "model"}
    return chosen


def update_trial(state, equity, now, eligible, drawdown=TRIAL_DRAWDOWN, days=TRIAL_DAYS):
    """Persistent latch; absence of valid equity never advances the high-water mark."""
    now = utc(now)
    state = dict(state)
    if state.get("stopped"):
        return state
    if equity is None or not math.isfinite(equity):
        return state
    if state.get("started_at") is None:
        if not eligible:
            return state
        state.update(started_at=now.isoformat(), peak_equity=equity)
    peak = number(state["peak_equity"], "peak equity")
    state["peak_equity"] = max(peak, equity)
    state["equity"] = equity
    if state["peak_equity"] - equity >= drawdown:
        state["stopped"] = "trial_drawdown"
    elif (now - utc(state["started_at"])).total_seconds() >= days * 86400:
        state["stopped"] = "trial_complete"
    return state
