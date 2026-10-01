"""Event-ordered public-data replay. It is an execution proxy, not a live queue simulation."""
import heapq
import itertools

import numpy as np
import pandas as pd

from quote_model import HORIZON, POSITION_STOP, liquidation, next_unlock, quote_decision, update_trial, validate_params
from utils import utc


def replay(book, trades, parameters, start, end, funding=None, latency=1., fee_extra=0.,
           all_taker=False, policy=quote_decision):
    """Start flat with a fresh risk state; latency delays order activation and cancellation."""
    start, end = utc(start), utc(end)
    if end <= start or latency < 0:
        raise ValueError("Invalid replay interval or latency")
    events, serial = [], itertools.count()

    def push(at, kind, value=None):
        if start.timestamp() <= at <= end.timestamp():
            heapq.heappush(events, (float(at), next(serial), kind, value))

    rows = book[(book.index >= start - pd.Timedelta(seconds=2)) & (book.index <= end)]
    for at, row in rows.iterrows():
        snapshot = {"timestamp": at.timestamp() * 1000, "received_at": row.received_at.timestamp(),
                    "bids": [], "asks": []}
        for side, key in (("bid", "bids"), ("ask", "asks")):
            for i in range(20):
                price, size = row.get(f"{side}_price_{i}"), row.get(f"{side}_size_{i}")
                if price is None or size is None or not np.isfinite([price, size]).all():
                    break
                snapshot[key].append([float(price), float(size)])
        # Seed only observations actually available before the replay starts.
        push(max(at.timestamp(), start.timestamp()), "book", snapshot)
        push(max(row.available_at.timestamp(), start.timestamp()), "received_book", snapshot)
    for at, row in trades[(trades.index >= start) & (trades.index <= end)].iterrows():
        push(at.timestamp(), "trade", (row.side, float(row.price), float(row["size"])))
    for p in sorted(parameters, key=lambda p: p["timestamp"]):
        push(max(utc(p["timestamp"]).timestamp(), start.timestamp()), "parameters", p)
    funding = funding if funding is not None else pd.DataFrame()
    funding_hours = set()
    if not funding.empty:
        for at, row in funding.iterrows():
            funding_hours.add(at.timestamp())
            push(at.timestamp(), "funding", (float(row.funding_rate), row.get("mark_price")))
    for at in pd.date_range(start.ceil("h"), end, freq="h"):
        if at.timestamp() not in funding_hours:
            push(at.timestamp(), "missing_funding")
    for at in pd.date_range(start, end, freq="15s", inclusive="left"):
        push(at.timestamp(), "loop")
    push(end.timestamp(), "terminal")
    cash, quantity, fees, funding_paid = 0., 0., 0., 0.
    order, physical, received, params = None, None, None, None
    params_valid = False
    first_fill, cycle_cash, cycle_notional, cooldown, closed_pending = None, 0., 0., 0., False
    fills, cycles, equity, errors = [], [], [], []
    state = {"started_at": None, "peak_equity": None, "stopped": None}
    pending_cancel = False

    def mark(at):
        if physical is None or at - physical["timestamp"] / 1000 > 2:
            return None
        if quantity <= 1e-12:
            return cash
        try:
            fee = (params["market"]["taker_fee"] if params else .00045) + fee_extra
            return cash + liquidation(physical, quantity, fee)[0]
        except ValueError:
            return None

    def fill(at, side, amount, price, taker):
        nonlocal cash, quantity, fees, first_fill, cycle_cash, cycle_notional, closed_pending
        if amount <= 1e-12:
            return
        if side == "buy" and quantity <= 1e-12:
            first_fill, cycle_cash, cycle_notional = at, cash, 0.
        if side == "buy":
            cycle_notional += amount * price
        fee = params["market"]["taker_fee" if taker or all_taker else "maker_fee"] + fee_extra
        cost = amount * price * fee
        fees += cost
        cash += (-amount * price - cost) if side == "buy" else (amount * price - cost)
        quantity += amount if side == "buy" else -amount
        fills.append({"time": at, "side": side, "quantity": amount, "price": price,
                      "fee": cost, "liquidity": "taker" if taker or all_taker else "maker"})
        if quantity < 1e-10:
            quantity = 0.
            cycles.append({"entry_time": first_fill, "exit_time": at, "pnl": cash - cycle_cash})
            first_fill, closed_pending = None, True

    def execute_marketable(at):
        nonlocal order
        if order is None or physical is None or at - physical["timestamp"] / 1000 > 2:
            return
        side = order["side"]
        levels = physical["asks" if side == "buy" else "bids"]
        for price, size in levels:
            if (side == "buy" and price > order["price"]) or (side == "sell" and price < order["price"]):
                break
            amount = min(order["remaining"], size, quantity if side == "sell" else order["remaining"])
            fill(at, side, amount, price, True)
            order["remaining"] -= amount
            if order["remaining"] < 1e-10:
                order = None
                break

    def decide(at):
        nonlocal order, state, cooldown, closed_pending
        now = pd.Timestamp(at, unit="s", tz="UTC")
        if closed_pending:
            cooldown, closed_pending = next_unlock(at), False
        value = mark(at)
        eligible = False
        if params_valid and params is not None:
            try:
                validate_params(params, params["pair"], now)
                eligible = True
            except (ValueError, KeyError, TypeError):
                pass
        state = update_trial(state, None if value is None else 1000 + value, now, eligible)
        if value is not None:
            equity.append({"time": at, "liquidation_pnl": value, "quantity": quantity,
                           "mid_pnl": cash + quantity * sum(physical[s][0][0] for s in ("bids", "asks")) / 2})
        protective = bool(state["stopped"] or not eligible)
        if quantity and first_fill is not None:
            protective |= at - first_fill >= HORIZON
            protective |= value is not None and value - cycle_cash <= -POSITION_STOP * cycle_notional
        if quantity == 0 and (not eligible or state["stopped"] or at < cooldown):
            return
        if received is None or at - received["timestamp"] / 1000 > 2:
            return
        try:
            if quantity and protective:
                price = liquidation(received, quantity, params["market"]["taker_fee"] + fee_extra)[1]
                decision = {"action": "liquidate", "price": price, "quantity": quantity}
            else:
                remaining = HORIZON if first_fill is None else HORIZON - (at - first_fill)
                decision = policy(params, received, now, quantity, remaining)
        except (ValueError, KeyError, TypeError):
            return
        if decision["action"] == "wait":
            return
        side = "buy" if decision["action"] == "buy" else "sell"
        order = {"side": side, "price": decision["price"], "remaining": decision["quantity"], "active": False}
        push(at + latency, "activate", order)

    while events:
        at, _, kind, item = heapq.heappop(events)
        if kind == "book":
            physical = item
        elif kind == "received_book":
            received = item
        elif kind == "parameters":
            params_valid = False
            try:
                validate_params(item, parameters[0]["pair"], pd.Timestamp(at, unit="s", tz="UTC"))
                params, params_valid = item, True
            except (ValueError, KeyError, TypeError):
                pass  # retain last costs for liquidation, never for new entries
        elif kind == "activate":
            if order is item:
                order["active"] = True
                execute_marketable(at)
        elif kind == "cancel":
            order, pending_cancel = None, False
            decide(at)
        elif kind == "loop":
            if (quantity or order is not None) and (physical is None or at - physical["timestamp"] / 1000 > 2):
                errors.append("Book gap during order/inventory exposure")
            if order is not None and not pending_cancel:
                # Keep the old order exposed until cancellation acknowledgement.
                pending_cancel = True
                push(at + latency, "cancel")
            elif order is None and not pending_cancel:
                decide(at)
        elif kind == "trade" and order is not None and order["active"]:
            side, price, amount = item
            crossed = ((order["side"] == "buy" and side == "sell" and price < order["price"]) or
                       (order["side"] == "sell" and side == "buy" and price > order["price"]))
            if crossed:
                amount = min(amount, order["remaining"], quantity if order["side"] == "sell" else amount)
                fill(at, order["side"], amount, order["price"], False)
                order["remaining"] -= amount
                if order["remaining"] < 1e-10:
                    order = None
        elif kind == "funding" and quantity:
            rate, observed_mark = item
            price = observed_mark if observed_mark is not None and np.isfinite(observed_mark) else (
                sum(physical[s][0][0] for s in ("bids", "asks")) / 2 if physical else None)
            if price is None:
                errors.append("Funding mark unavailable")
            else:
                charge = quantity * price * rate
                cash -= charge
                funding_paid += charge
        elif kind == "missing_funding" and quantity:
            errors.append("Settled funding observation missing")
        elif kind == "terminal":
            order = None
            value = mark(at)
            if value is None:
                errors.append("Terminal liquidation unobservable")
            else:
                if quantity:
                    proceeds, _ = liquidation(physical, quantity, params["market"]["taker_fee"] + fee_extra)
                    rate = proceeds / quantity / (1 - params["market"]["taker_fee"] - fee_extra)
                    fill(at, "sell", quantity, rate, True)
                equity.append({"time": at, "liquidation_pnl": cash, "mid_pnl": cash, "quantity": quantity})
    # Markouts require a fresh subsequent observation; no interpolation across missing time.
    bt = book.index.astype("int64").to_numpy() / 1e9
    mids = book.mid_price.to_numpy()
    for item in fills:
        for delay in (1, 5, 15, 60):
            when = item["time"] + delay
            idx = int(np.searchsorted(bt, when, side="right") - 1)
            item[f"markout_{delay}s"] = (float((mids[idx] - item["price"]) * item["quantity"] *
                                             (1 if item["side"] == "buy" else -1))
                                        if idx >= 0 and 0 <= when - bt[idx] <= 2 and when <= end.timestamp() else None)
    series = np.array([r["liquidation_pnl"] for r in equity])
    max_dd = float(np.max(np.maximum.accumulate(np.r_[0., series])[1:] - series)) if len(series) else 0.
    return {"valid": not errors, "errors": sorted(set(errors)), "net_pnl": cash if not quantity else None,
            "fees": fees, "funding_paid": funding_paid, "round_trips": len(cycles),
            "max_drawdown": max_dd, "fills": fills, "cycles": cycles, "equity": equity,
            "latency_seconds": latency, "trial_state": state}
