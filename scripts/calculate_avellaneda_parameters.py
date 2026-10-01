"""Estimate the current constrained policy from a causal window; never retain a failed entry gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from intensity import estimate_intensity, window_depths
from quote_model import GAMMA_USDC, HORIZON, STAKE, validate_params
from utils import atomic_json, get_tick_size, load_book_data, load_trades_data, mid_grid, safe_read_parquet, stream_path, utc
from volatility import forecast_variance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "user_data" / "strategies"))
from pair_loader import get_active_pair


def model_digest():
    digest = hashlib.sha256()
    for name in ("quote_model.py", "volatility.py", "intensity.py", "backtest.py", "utils.py", "calculate_avellaneda_parameters.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def estimate(ticker, data_root, asof=None, bootstrap=32, observations=None):
    live = asof is None
    asof = utc(asof) if asof is not None else pd.Timestamp.now(tz="UTC")
    pair = f"{ticker}/USDC:USDC"
    start = asof - pd.Timedelta(hours=24)
    p = {"pair": pair, "timestamp": asof.isoformat(), "data_valid": False, "trade_enabled": False,
         "reasons": [], "gamma_usdc": GAMMA_USDC, "model_digest": model_digest(),
         "evidence": {"status": "inconclusive", "reason": "No accepted chronological evaluation"}}
    def cut(frame, beginning, cutoff):
        dates = frame.index if isinstance(frame.index, pd.DatetimeIndex) else frame.event_time
        return frame[(dates >= beginning) & (dates <= cutoff) & (frame.available_at <= cutoff)]

    book = (load_book_data(stream_path(data_root, ticker, "orderbooks"), start, asof, depth=1)
            if observations is None else cut(observations["book"], start, asof))
    if book.empty:
        raise ValueError("No current book observations")
    data_end = book.index.max()
    data_start = max(start, book.index.min())
    p.update(data_start=data_start.isoformat(), data_end=data_end.isoformat(),
             expires_at=(data_end + pd.Timedelta(minutes=30)).isoformat())
    if (asof - data_end).total_seconds() > 600:
        p["reasons"].append("Book capture is more than ten minutes old")
    if (data_end - data_start).total_seconds() < 6 * 3600:
        p["reasons"].append("Need six hours of observations")
    p["rejected_clocks"] = {"orderbooks": int((~book.clock_valid).sum())}
    if not book.clock_valid.iloc[-1]:
        p["reasons"].append("Latest book clock invalid")
    grid_start, grid_end = data_start.ceil("5s"), data_end.floor("5s")
    mid = mid_grid(book, grid_start, grid_end)
    p["coverage"] = float(mid.notna().mean()) if len(mid) else 0.
    if p["coverage"] < .95:
        p["reasons"].append("Book coverage below 95%")
    candidates = list((Path(data_root) / f"market_{ticker}").glob("*.json"))
    candidates.append(Path(data_root) / f"market_{ticker}.json")
    known = []
    for path in candidates:
        if path.exists():
            item = json.loads(path.read_text())
            if utc(item["timestamp"]) <= asof:
                known.append(item)
    if not known:
        raise ValueError("No market metadata known at the requested cutoff")
    market = max(known, key=lambda m: m["timestamp"])
    p["market"] = market
    reference = float(book.mid_price.iloc[-1])
    p["reference_mid"] = reference
    unit = 10.0 ** -market["sz_decimals"]
    quantity = round(np.floor(STAKE / reference / unit) * unit, market["sz_decimals"])
    if quantity <= 0:
        raise ValueError("Stake is below one quantity increment")
    p["quantity"] = quantity
    p["volatility"] = forecast_variance(mid)
    p["volatility"]["origin"] = grid_end.isoformat()
    trades = (load_trades_data(stream_path(data_root, ticker, "trades"), data_start, asof)
              if observations is None else cut(observations["trades"], data_start, asof))
    p["rejected_clocks"]["trades"] = int((~trades.clock_valid).sum())
    valid_trades = trades[trades.clock_valid]
    if valid_trades.empty:
        raise ValueError("No captured trades with valid clocks")
    p["trade_data_end"] = valid_trades.index.max().isoformat()
    p["expires_at"] = (min(grid_end, valid_trades.index.max()) + pd.Timedelta(minutes=30)).isoformat()
    if (asof - valid_trades.index.max()).total_seconds() > 600:
        p["reasons"].append("Trade capture is more than ten minutes old")
    windows = window_depths(book, trades, data_start, data_end, quantity)
    p["intensity_coverage"] = float(windows.attrs["coverage"])
    if p["intensity_coverage"] < .95:
        p["reasons"].append("Quote exposure coverage below 95%")
    tick = get_tick_size(reference, market["sz_decimals"])
    p["intensity"] = {}
    for side in ("bid", "ask"):
        model = estimate_intensity(windows[f"{side}_depth"], tick, bootstrap=bootstrap)
        model["quantity"] = quantity
        p["intensity"][side] = model
    context = (safe_read_parquet(stream_path(data_root, ticker, "contexts"), asof - pd.Timedelta(minutes=10), asof)
               if observations is None else cut(observations["contexts"], asof - pd.Timedelta(minutes=10), asof))
    if context.empty:
        raise ValueError("No recent funding context")
    p["funding_rate"] = float(context.funding_rate.iloc[-1])
    if not np.isfinite(p["funding_rate"]):
        raise ValueError("Invalid funding rate")
    health_file = Path(data_root) / "health.json"
    if live and not health_file.exists():
        p["reasons"].append("Collector health unavailable")
    elif live:
        health = json.loads(health_file.read_text())
        age = (pd.Timestamp.now(tz="UTC") - utc(health["timestamp"])).total_seconds()
        if not -1 <= age <= 30 or not health.get("symbols", {}).get(ticker, {}).get("healthy"):
            p["reasons"].append("Collector is not healthy")
    p["data_valid"] = not p["reasons"]
    if p["data_valid"]:
        validate_params(p, pair, asof)
    return p


def apply_evidence(p, path):
    if path is None or not Path(path).exists():
        return p
    evidence = json.loads(Path(path).read_text())
    if (evidence.get("pair") == p["pair"] and evidence.get("model_digest") == p["model_digest"] and
            evidence.get("status") == "passed" and evidence.get("days", 0) >= 7 and
            evidence.get("round_trips", 0) >= 100 and evidence.get("lower_bound", 0) > 0 and
            evidence.get("stress_lower_bound", 0) > 0 and evidence.get("two_day_lower_bound", 0) > 0 and
            evidence.get("net_pnl", 0) > 0 and utc(evidence["data_end"]) <= utc(p["timestamp"])):
        p["evidence"] = evidence
        p["trade_enabled"] = p["data_valid"]
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--data-dir", default=os.getenv("HL_DATA_LOC", str(ROOT / "HL_data_collector/HL_data")))
    parser.add_argument("--output-dir", default=os.getenv("AVELLANEDA_PARAMS_DIR", str(ROOT / "scripts")))
    parser.add_argument("--asof", help="UTC cutoff for offline diagnostics; requires historically available metadata")
    parser.add_argument("--evidence", default=os.getenv("AVELLANEDA_EVIDENCE"))
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    ticker = (args.ticker or get_active_pair().split("/")[0]).upper()
    while True:
        started = time.monotonic()
        try:
            result = apply_evidence(estimate(ticker, args.data_dir, args.asof), args.evidence)
        except Exception as exc:
            result = {"pair": f"{ticker}/USDC:USDC", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                      "data_valid": False, "trade_enabled": False, "reasons": [str(exc)],
                      "evidence": {"status": "inconclusive"}}
        result["calculation_seconds"] = time.monotonic() - started
        result["estimation_asof"] = result["timestamp"]
        if args.asof is None:
            result["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
            if result["data_valid"]:
                try:
                    validate_params(result, result["pair"], utc(result["timestamp"]))
                except (ValueError, KeyError, TypeError) as exc:
                    result.update(data_valid=False, trade_enabled=False, reasons=[str(exc)])
        atomic_json(Path(args.output_dir) / f"avellaneda_parameters_{ticker}.json", result)
        if result["data_valid"]:
            stamp = utc(result["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
            atomic_json(Path(args.output_dir) / ticker / f"{stamp}.json", result)
        print(json.dumps({"pair": result["pair"], "data_valid": result["data_valid"],
                          "trade_enabled": result["trade_enabled"], "reasons": result.get("reasons", []),
                          "volatility": result.get("volatility", {}).get("diagnostics")}), flush=True)
        if not args.loop:
            return
        time.sleep(max(1, (900 if result["data_valid"] else 60) - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
