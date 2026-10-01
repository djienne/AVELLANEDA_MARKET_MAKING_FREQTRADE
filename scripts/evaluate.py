"""Chronological policy evaluation with liquidation accounting and explicit insufficient-data results."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from backtest import replay
from calculate_avellaneda_parameters import estimate, model_digest
from quote_model import liquidation
from utils import atomic_json, load_book_data, load_funding_data, load_trades_data, safe_read_parquet, stream_path, utc


def lower_bound(values, block_days=1):
    """One-sided 95% circular block-bootstrap bound on mean PnL per usable day."""
    values = np.asarray(values, float)
    if len(values) < 7:
        return None
    rng = np.random.default_rng(100)
    means = []
    for _ in range(2000):
        starts = rng.integers(0, len(values), int(np.ceil(len(values) / block_days)))
        idx = (starts[:, None] + np.arange(block_days)) % len(values)
        means.append(values[idx.ravel()[:len(values)]].mean())
    return float(np.quantile(means, .05))


def fixed_quote(p, book, now, quantity, remaining):
    mid = sum(book[s][0][0] for s in ("bids", "asks")) / 2
    if quantity and remaining <= 0:
        return {"action": "liquidate", "price": liquidation(book, quantity, p["market"]["taker_fee"])[1],
                "quantity": quantity}
    side = "ask" if quantity else "bid"
    delta = max(p["intensity"][side]["min_delta"], mid * .0005)
    if delta > p["intensity"][side]["max_delta"]:
        return {"action": "wait", "price": None, "quantity": quantity}
    return {"action": "sell" if quantity else "buy", "price": mid + (delta if quantity else -delta),
            "quantity": quantity or p["quantity"]}


def evaluate(data_root, ticker, start, end, output):
    """Refit chronologically; each replay day resets inventory and the simulated risk budget."""
    start, end = utc(start), utc(end)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"pair": f"{ticker}/USDC:USDC", "model_digest": model_digest(),
               "data_start": start.isoformat(), "data_end": end.isoformat(),
               "status": "inconclusive", "days": 0, "round_trips": 0, "reasons": [], "daily": []}
    scenarios = {"base": {"latency": 1.}, "zero_latency": {"latency": 0.},
                 "slow": {"latency": 5.}, "higher_fees": {"latency": 1., "fee_extra": .0001},
                 "all_taker": {"latency": 1., "all_taker": True},
                 "fixed_quote": {"latency": 1., "policy": fixed_quote}}
    plotted, last_params = [], None
    for beginning in pd.date_range(start, end, freq="D", inclusive="left"):
        finish = min(beginning + pd.Timedelta(days=1), end)
        prior = beginning - pd.Timedelta(hours=24)
        try:
            observed = {"book": load_book_data(stream_path(data_root, ticker, "orderbooks"), prior, finish),
                        "trades": load_trades_data(stream_path(data_root, ticker, "trades"), prior, finish),
                        "contexts": safe_read_parquet(stream_path(data_root, ticker, "contexts"), prior, finish)}
            try:
                funding = load_funding_data(stream_path(data_root, ticker, "funding"), beginning, finish)
            except ValueError:
                funding = pd.DataFrame()
            schedule = []
            problems = []
            for cutoff in pd.date_range(beginning, finish, freq="15min", inclusive="left"):
                began = time.monotonic()
                try:
                    p = estimate(ticker, data_root, cutoff, bootstrap=0, observations=observed)
                    if not p["data_valid"]:
                        problems.extend(p["reasons"])
                    else:
                        last_params = p
                except Exception as exc:
                    problems.append(str(exc))
                    p = {"pair": summary["pair"], "timestamp": cutoff.isoformat(),
                         "data_valid": False, "trade_enabled": False, "reasons": [str(exc)]}
                p["estimation_asof"] = cutoff.isoformat()
                p["calculation_seconds"] = time.monotonic() - began
                delay = max(15, math.ceil(p["calculation_seconds"] / 15) * 15)
                p["timestamp"] = (cutoff + pd.Timedelta(seconds=delay)).isoformat()
                schedule.append(p)
            if problems:
                summary["reasons"].extend(problems)
            day = {"start": beginning.isoformat(), "end": finish.isoformat(),
                   "estimates_valid": not problems, "scenarios": {}}
            for name, options in scenarios.items():
                result = replay(observed["book"], observed["trades"], schedule, beginning, finish,
                                funding=funding, **options)
                atomic_json(output / f"{beginning.strftime('%Y%m%d')}_{name}.json", result)
                day["scenarios"][name] = {k: result[k] for k in
                    ("valid", "errors", "net_pnl", "round_trips", "fees", "funding_paid", "max_drawdown")}
                if name == "base":
                    plotted.extend(result["equity"])
            day["usable"] = bool(not problems and finish - beginning == pd.Timedelta(days=1) and
                                 all(v["valid"] for v in day["scenarios"].values()))
            summary["daily"].append(day)
        except Exception as exc:
            summary["reasons"].append(f"{beginning.date()}: {exc}")
    usable = [day for day in summary["daily"] if day["usable"]]
    summary["days"] = len(usable)
    summary["round_trips"] = sum(day["scenarios"]["base"]["round_trips"] for day in usable)
    base = [day["scenarios"]["base"]["net_pnl"] for day in usable]
    slow = [day["scenarios"]["slow"]["net_pnl"] for day in usable]
    summary.update(lower_bound=lower_bound(base), stress_lower_bound=lower_bound(slow),
                   two_day_lower_bound=lower_bound(base, 2), net_pnl=float(sum(base)), no_trading_pnl=0.)
    if len(usable) >= 7 and summary["round_trips"] >= 100:
        passed = (sum(base) > 0 and sum(slow) > 0 and summary["lower_bound"] > 0 and
                  summary["stress_lower_bound"] > 0 and summary["two_day_lower_bound"] > 0)
        summary["status"] = "passed" if passed else "negative"
    else:
        summary["reasons"].append("Need seven complete evaluation days and 100 completed round trips")
    summary["reasons"] = sorted(set(summary["reasons"]))
    atomic_json(output / "evidence.json", summary)
    write_plots(output, summary, plotted, last_params)
    return summary


def write_plots(output, summary, equity, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(f"{summary['pair']} — public-data replay: {summary['status']}")
    if equity:
        df = pd.DataFrame(equity)
        axes[0, 0].plot(pd.to_datetime(df.time, unit="s", utc=True), df.liquidation_pnl)
        axes[0, 1].plot(pd.to_datetime(df.time, unit="s", utc=True), df.quantity)
    axes[0, 0].set(title="Liquidation PnL (each evaluation day starts flat)", ylabel="USDC")
    axes[0, 1].set(title="Inventory", ylabel="Base units")
    if params:
        for side in ("bid", "ask"):
            cal = params["intensity"][side]["calibration"]
            axes[1, 0].plot(cal["delta"], cal["observed"], "o", label=side + " observed")
            axes[1, 0].plot(cal["delta"], cal["predicted"], "-", label=side + " fitted")
        curve = params["volatility"]["cumulative_log_variance"]
        axes[1, 1].plot(np.arange(len(curve)) * 5 / 60, curve)
        axes[1, 0].legend()
    axes[1, 0].set(title="Full-quantity crossing proxy", xlabel="Distance (USDC/base unit)", ylabel="Probability / 15 s")
    axes[1, 1].set(title="Cumulative forecast variance", xlabel="Minutes", ylabel="Log-return variance")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.data_dir, args.ticker.upper(), args.start, args.end, args.output)
    print(json.dumps({k: result[k] for k in ("status", "days", "round_trips", "net_pnl", "reasons")}, indent=2))
