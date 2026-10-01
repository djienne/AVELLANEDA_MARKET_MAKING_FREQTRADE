"""Docker benchmark/parity check against a saved pre-optimization scripts directory."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import pandas as pd

from backtest import prepare_replay_events, replay
from quote_model import warm_quote_kernel
from utils import atomic_json, load_book_data, load_funding_data, load_trades_data, stream_path, utc


def load_reference(directory):
    """Load the old modules without maintaining a second production implementation."""
    names = ("utils", "quote_model", "backtest")
    current = {name: sys.modules[name] for name in names}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, Path(directory) / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules["backtest"], sys.modules["quote_model"]
    finally:
        sys.modules.update(current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "schedule", "reference-dir", "start", "end", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ticker", default="PAXG")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    start, end = utc(args.start), utc(args.end)
    began = time.perf_counter()
    book = load_book_data(stream_path(args.data_dir, args.ticker, "orderbooks"),
                          start - pd.Timedelta(seconds=2), end)
    trades = load_trades_data(stream_path(args.data_dir, args.ticker, "trades"), start, end)
    funding = load_funding_data(stream_path(args.data_dir, args.ticker, "funding"), start, end)
    schedule = [p for p in json.loads(Path(args.schedule).read_text())
                if start <= utc(p["estimation_asof"]) < end]
    report = {"start": start.isoformat(), "end": end.isoformat(), "book_rows": len(book),
              "trades": len(trades), "load_seconds": time.perf_counter() - began}
    reference, reference_quotes = load_reference(args.reference_dir)
    began = time.perf_counter()
    warm_quote_kernel()
    report["kernel_first_call_seconds"] = time.perf_counter() - began
    began = time.perf_counter()
    warm_quote_kernel()
    report["kernel_warm_call_seconds"] = time.perf_counter() - began
    began = time.perf_counter()
    prepared = prepare_replay_events(book, trades, schedule, start, end, funding)
    report["prepare_seconds"] = time.perf_counter() - began
    inputs = (book, trades, schedule, start, end)
    variants = {"reference": (reference.replay, {}),
                "arrays_only": (replay, {"policy": reference_quotes.quote_decision}),
                "arrays_numba": (replay, {}),
                "prepared_numba": (replay, {"prepared_events": prepared})}
    report["runs_seconds"] = {name: [] for name in variants}
    expected = None
    # Interleave variants to reduce bias from concurrent machine activity.
    for _ in range(args.repeats):
        for name, (run, options) in variants.items():
            began = time.perf_counter()
            result = run(*inputs, funding=funding, **options)
            report["runs_seconds"][name].append(time.perf_counter() - began)
            if expected is None:
                expected = result
            assert result == expected, f"Replay behavior differs: {name}"
    report["median_seconds"] = {name: statistics.median(values)
                                for name, values in report["runs_seconds"].items()}
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["ledger_parity"] = True
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
