"""Small scientific and execution regressions. Run with the project's Docker image."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "user_data/strategies"), str(ROOT / "HL_data_collector")]
from backtest import replay
from hyperliquid_data_collector import HyperliquidDataCollector
from intensity import fit_maxima, window_depths
from quote_model import next_unlock, quote_decision, update_trial, validate_params
from utils import atomic_json, load_funding_data, load_trades_data
from volatility import forecast_variance


def parameters(now):
    now = pd.Timestamp(now)
    return {"pair": "PAXG/USDC:USDC", "timestamp": (now - pd.Timedelta(seconds=2)).isoformat(),
            "data_start": (now - pd.Timedelta(hours=24)).isoformat(),
            "data_end": (now - pd.Timedelta(seconds=10)).isoformat(),
            "trade_data_end": (now - pd.Timedelta(seconds=10)).isoformat(),
            "expires_at": (now + pd.Timedelta(minutes=20)).isoformat(), "coverage": 1.,
            "data_valid": True, "trade_enabled": False, "evidence": {"status": "inconclusive"},
            "reference_mid": 100., "quantity": .5, "gamma_usdc": 2., "funding_rate": 0.,
            "market": {"symbol": "PAXG/USDC:USDC", "timestamp": (now - pd.Timedelta(hours=1)).isoformat(),
                       "sz_decimals": 4, "maker_fee": .00015, "taker_fee": .00045,
                       "min_notional": 10., "min_amount": .0001},
            "volatility": {"sample_seconds": 5, "origin": (now - pd.Timedelta(seconds=10)).isoformat(),
                           "cumulative_log_variance": (np.arange(721) * 1e-8).tolist()},
            "intensity": {s: {"A": .1, "k": 10., "min_delta": .1, "max_delta": .8,
                             "windows": 2000, "crossings": 100, "quantity": .5} for s in ("bid", "ask")}}


def live_book(now, mid=100.):
    return {"timestamp": now.timestamp() * 1000 - 100, "received_at": now.timestamp() - .05,
            "bids": [[mid - .01, 10.]], "asks": [[mid + .01, 10.]]}


def recorded(start, seconds=60, final_mid=100.):
    times = pd.date_range(start - pd.Timedelta(seconds=2), start + pd.Timedelta(seconds=seconds), freq="s")
    mid = np.full(len(times), 100.)
    mid[-2:] = final_mid
    book = pd.DataFrame({"mid_price": mid, "bid_price_0": mid - .01, "ask_price_0": mid + .01,
                         "bid_size_0": 10., "ask_size_0": 10., "clock_valid": True,
                         "received_at": times + pd.Timedelta(milliseconds=10),
                         "available_at": times + pd.Timedelta(milliseconds=10)}, index=times)
    book.index.name = "event_time"
    return book


class Checks(unittest.TestCase):
    def test_parameter_boundary(self):
        now = pd.Timestamp.now(tz="UTC")
        p = parameters(now)
        validate_params(p, p["pair"], now)
        for key, value in [("data_valid", "true"), ("gamma_usdc", 0.), ("quantity", float("nan")),
                           ("data_end", (now + pd.Timedelta(days=7)).isoformat()),
                           ("pair", "ETH/USDC:USDC")]:
            bad = copy.deepcopy(p)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError, TypeError)):
                validate_params(bad, p["pair"], now)

    def test_garch_failure_uses_fallback(self):
        now = pd.Timestamp("2026-10-01", tz="UTC")
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 1e-4, 1000))),
                           index=pd.date_range(now, periods=1000, freq="5s"))
        failed = SimpleNamespace(params={"omega": .1, "alpha[1]": .1, "beta[1]": .8, "nu": 5},
                                 convergence_flag=4,
                                 optimization_result=SimpleNamespace(success=False, message="failed"),
                                 forecast=Mock(side_effect=AssertionError("Must not forecast a failed fit")))
        with patch("volatility.arch_model", return_value=SimpleNamespace(fit=lambda **kw: failed)):
            result = forecast_variance(prices)
        self.assertEqual(result["diagnostics"]["method"], "ewma")
        self.assertEqual(result["diagnostics"]["convergence_flag"], 4)
        failed.forecast.assert_not_called()
        self.assertTrue(np.isfinite(result["cumulative_log_variance"]).all())

    def test_sigma_known_truth_and_no_future(self):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2026-10-01", periods=6000, freq="5s", tz="UTC")
        sig = .025
        mid = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, sig * np.sqrt(5 / 86400), len(idx)))), index=idx)
        first = forecast_variance(mid.iloc[:4000])
        altered = mid.copy()
        altered.iloc[4000:] *= 2
        again = forecast_variance(altered.iloc[:4000])
        self.assertEqual(first, again)
        self.assertLess(abs(first["sigma_daily"] / sig - 1), .3)

    def test_no_returns_across_gaps(self):
        mid = pd.Series(np.r_[np.full(600, 100.), np.nan, np.full(600, 10000.)])
        result = forecast_variance(mid)
        self.assertEqual(result["sigma_daily"], 0.)
        self.assertEqual(result["diagnostics"]["method"], "ewma")

    def test_intensity_known_truth(self):
        rng = np.random.default_rng(42)
        a, k, dt, tick = .08, 2.5, 15, .1
        depths = np.maximum(0, -np.log(-np.log(rng.uniform(size=6000)) / (a * dt)) / k)
        estimated = fit_maxima(depths, tick, dt)
        np.testing.assert_allclose(estimated, [a, k], rtol=.12)

    def test_print_fragmentation(self):
        start = pd.Timestamp("2026-10-01T12:01:00Z")
        book = recorded(start, 60)
        at = start + pd.Timedelta(seconds=5)
        def trades(n):
            return pd.DataFrame({"price": 99.5, "size": .5 / n, "side": "sell", "clock_valid": True},
                                index=pd.DatetimeIndex([at] * n))
        one = window_depths(book, trades(1), start, start + pd.Timedelta(seconds=60), .5)
        split = window_depths(book, trades(4), start, start + pd.Timedelta(seconds=60), .5)
        pd.testing.assert_frame_equal(one, split)

    def test_bad_trade_clock_excludes_both_exposure_windows(self):
        start = pd.Timestamp("2026-10-01T12:01:00Z")
        bad = pd.DataFrame({"price": [99.5], "size": [.5], "side": ["sell"], "clock_valid": [False],
                            "received_at": [start + pd.Timedelta(seconds=14)]},
                           index=[start + pd.Timedelta(seconds=16)])
        windows = window_depths(recorded(start, 60), bad, start, start + pd.Timedelta(seconds=60), .5)
        self.assertEqual(list(windows.index), [start + pd.Timedelta(seconds=s) for s in (30, 45)])
        self.assertEqual(windows.attrs["coverage"], .5)

    def test_dynamic_program_one_step(self):
        now = pd.Timestamp("2026-10-01T12:01:00Z")
        p, b = parameters(now), live_book(now)
        p["volatility"]["cumulative_log_variance"] = [0.] * 721
        p["market"]["maker_fee"] = p["market"]["taker_fee"] = 0.
        for side in p["intensity"].values():
            side["max_delta"] = side["min_delta"] = .1
        out = quote_decision(p, b, now, planning_seconds=15)
        probability = -np.expm1(-.1 * np.exp(-10 * .1) * 15)
        expected = probability * (.5 * .1 - .5 * .01)
        self.assertEqual(out["action"], "buy")
        self.assertAlmostEqual(out["value"], expected, places=9)

    def test_unit_scaling_and_deadline(self):
        now = pd.Timestamp.now(tz="UTC")
        p, b = parameters(now), live_book(now)
        first = quote_decision(p, b, now)
        scaled = copy.deepcopy(p)
        scaled["quantity"] /= 10
        scaled["reference_mid"] *= 10
        for side in scaled["intensity"].values():
            side["k"] /= 10
            side["min_delta"] *= 10
            side["max_delta"] *= 10
        larger = {"timestamp": b["timestamp"], "received_at": b["received_at"],
                  "bids": [[b["bids"][0][0] * 10, 10]], "asks": [[b["asks"][0][0] * 10, 10]]}
        second = quote_decision(scaled, larger, now)
        self.assertEqual(first["action"], second["action"])
        self.assertAlmostEqual(first["value"], second["value"], places=7)
        close = quote_decision(p, b, now, quantity=.1, remaining_seconds=0)
        self.assertEqual(close["action"], "liquidate")
        self.assertEqual(close["quantity"], .1)

    def test_dynamic_program_three_step_enumeration(self):
        now = pd.Timestamp("2026-10-01T12:01:00Z")
        p, b = parameters(now), live_book(now)
        p["market"]["maker_fee"] = p["market"]["taker_fee"] = 0.
        for side in p["intensity"].values():
            side["min_delta"] = side["max_delta"] = .1
        arrival = .1 * np.exp(-1)
        probability = -np.expm1(-arrival * 15)
        occupied = probability / (arrival * 15)
        risk = .5 * 2 * .5 ** 2 * 100 ** 2 * 3e-8
        # Enumerate both branches of every fill/no-fill action; an exit locks until after this horizon.
        def value(n, long):
            if n == 0:
                return -.005 if long else 0.
            wait = value(n - 1, long) - (risk if long else 0)
            if long:
                sell = probability * .05 + (1 - probability) * value(n - 1, True) - risk * occupied
                return max(-.005, wait, sell)
            buy = probability * (.05 + value(n - 1, True)) + (1 - probability) * value(n - 1, False)
            return max(wait, buy - risk * (1 - occupied))
        actual = quote_decision(p, b, now, planning_seconds=45)
        self.assertAlmostEqual(actual["value"], value(3, False), places=9)

    def test_order_remains_exposed_until_cancel_ack(self):
        start = pd.Timestamp("2026-10-01T12:01:00Z")
        book = recorded(start, 60)
        trades = pd.DataFrame({"side": ["sell", "sell"], "price": [99.8, 99.8], "size": [.2, .2]},
                              index=[start + pd.Timedelta(seconds=17), start + pd.Timedelta(seconds=22)])
        def policy(p, b, now, q, remaining):
            return {"action": "buy" if q == 0 else "wait", "price": 99.9, "quantity": .5}
        out = replay(book, trades, [parameters(start)], start, start + pd.Timedelta(seconds=60),
                     latency=5, policy=policy)
        buys = [f for f in out["fills"] if f["side"] == "buy"]
        self.assertEqual(len(buys), 1)
        self.assertAlmostEqual(buys[0]["quantity"], .2)

    def test_invalid_update_liquidates_existing_position(self):
        start = pd.Timestamp("2026-10-01T12:01:00Z")
        book = recorded(start, 60)
        trades = pd.DataFrame({"side": ["sell"], "price": [99.8], "size": [.5]},
                              index=[start + pd.Timedelta(seconds=3)])
        bad = {"pair": "PAXG/USDC:USDC", "timestamp": (start + pd.Timedelta(seconds=14)).isoformat(),
               "data_valid": False, "trade_enabled": False, "reasons": ["collector failure"]}
        def policy(p, b, now, q, remaining):
            return {"action": "buy" if q == 0 else "wait", "price": 99.9, "quantity": .5}
        out = replay(book, trades, [parameters(start), bad], start, start + pd.Timedelta(seconds=60), policy=policy)
        self.assertTrue(out["valid"], out["errors"])
        exits = [f for f in out["fills"] if f["side"] == "sell"]
        self.assertEqual(len(exits), 1)
        self.assertLess(exits[0]["time"], (start + pd.Timedelta(seconds=30)).timestamp())

    def test_terminal_mark_and_partial_volume(self):
        start = pd.Timestamp("2026-10-01T12:01:00Z")
        book = recorded(start, 60, 99.)
        trades = pd.DataFrame({"side": ["sell", "sell"], "price": [99.8, 99.8], "size": [.2, .1]},
                              index=[start + pd.Timedelta(seconds=3), start + pd.Timedelta(seconds=4)])
        def policy(p, b, now, q, remaining):
            return {"action": "buy" if q == 0 else "wait", "price": 99.9, "quantity": .5}
        out = replay(book, trades, [parameters(start)], start, start + pd.Timedelta(seconds=60), policy=policy)
        self.assertTrue(out["valid"], out["errors"])
        self.assertLess(out["net_pnl"], 0)
        self.assertAlmostEqual(sum(f["quantity"] for f in out["fills"] if f["side"] == "buy"), .3)
        self.assertEqual(out["round_trips"], 1)
        expected = -.3 * 99.9 * 1.00015 + .3 * 98.99 * .99955
        self.assertAlmostEqual(out["net_pnl"], expected)

    def test_framework_lock(self):
        from freqtrade.persistence import PairLocks
        from freqtrade.freqtradebot import FreqtradeBot
        from avellaneda import avellaneda
        PairLocks.use_db = False
        PairLocks.timeframe = "15m"
        PairLocks.reset_locks()
        now = pd.Timestamp("2026-10-01T12:01:00Z").to_pydatetime()
        strategy = avellaneda({})
        bot = SimpleNamespace(strategy=strategy, protections=SimpleNamespace(
            stop_per_pair=lambda *a, **kw: None, global_stop=lambda *a, **kw: None))
        with patch("freqtrade.freqtradebot.datetime") as dt:
            dt.now.return_value = now
            FreqtradeBot.handle_protections(bot, "PAXG/USDC:USDC", "long")
        lock = PairLocks.get_pair_longest_lock("PAXG/USDC:USDC", now, "long")
        self.assertEqual(lock.lock_end_time.hour, 12)
        self.assertEqual(lock.lock_end_time.minute, 15)
        self.assertEqual(next_unlock(now.timestamp()), lock.lock_end_time.replace(tzinfo=now.tzinfo).timestamp())

    def test_risk_latch_survives_restart(self):
        now = pd.Timestamp.now(tz="UTC")
        state = {"started_at": None, "peak_equity": None, "stopped": None}
        state = update_trial(state, 1000., now, True)
        state = update_trial(state, 1002., now, True)
        state = update_trial(state, 991.9, now, True)
        self.assertEqual(state["stopped"], "trial_drawdown")
        restored = json.loads(json.dumps(state))
        self.assertEqual(update_trial(restored, 1100, now, True)["stopped"], "trial_drawdown")

    def test_complete_estimator_and_atomic_output(self):
        from calculate_avellaneda_parameters import estimate
        rng = np.random.default_rng(21)
        cutoff = pd.Timestamp.now(tz="UTC").floor("15s")
        times = pd.date_range(end=cutoff - pd.Timedelta(seconds=1), periods=22000, freq="s")
        seconds = times.astype("int64").to_numpy() / 1e9
        mid = 100 * np.exp(np.cumsum(rng.normal(0, .025 / np.sqrt(86400), len(times))))
        book = pd.DataFrame({"timestamp": seconds + .02, "exchange_timestamp": (seconds * 1000).astype("int64"),
                             "bid_price_0": mid - .01, "ask_price_0": mid + .01,
                             "bid_size_0": 10., "ask_size_0": 10., "symbol": "PAXG"})
        prints = []
        for side, sign, offset in (("buy", 1, 5), ("sell", -1, 7)):
            indices = np.arange(offset, len(times), 15)
            prints.append(pd.DataFrame({"timestamp": seconds[indices] + .2,
                "exchange_timestamp": (seconds[indices] * 1000 + 100).astype("int64"),
                "price": mid[indices] + sign * rng.exponential(.3, len(indices)), "size": 1.,
                "side": side, "symbol": "PAXG", "trade_id": [f"{side}{i}" for i in indices]}))
        with tempfile.TemporaryDirectory() as td:
            book.loc[100, "timestamp"] -= 3  # isolated bad clocks must be excluded, not accepted or permanently latched
            book.to_parquet(Path(td) / "orderbooks_PAXG.parquet")
            prints[0].loc[100, "timestamp"] -= 3
            pd.concat(prints).to_parquet(Path(td) / "trades_PAXG.parquet")
            context = pd.DataFrame({"timestamp": [seconds[-1] + .02], "exchange_timestamp": [int(seconds[-1]*1000)],
                                    "funding_rate": [.00001], "mark_price": [mid[-1]], "symbol": ["PAXG"]})
            context.to_parquet(Path(td) / "contexts_PAXG.parquet")
            atomic_json(Path(td) / "market_PAXG.json", parameters(cutoff)["market"])
            result = estimate("PAXG", td, cutoff, bootstrap=0)
            self.assertTrue(result["data_valid"], result["reasons"])
            self.assertEqual(result["rejected_clocks"], {"orderbooks": 1, "trades": 1})
            validate_params(result, result["pair"], cutoff)
            self.assertFalse(result["trade_enabled"])
            self.assertGreater(result["volatility"]["cumulative_log_variance"][-1], 0)
            atomic_json(Path(td) / "parameters.json", result)
            self.assertEqual(json.loads((Path(td) / "parameters.json").read_text())["quantity"], result["quantity"])
            atomic_json(Path(td) / "health.json", {"timestamp": (cutoff + pd.Timedelta(seconds=9)).isoformat(),
                                                   "symbols": {"PAXG": {"healthy": True}}})
            clock = Mock(wraps=pd)
            clock.Timestamp.now.side_effect = [cutoff, cutoff + pd.Timedelta(seconds=10)]
            with patch("calculate_avellaneda_parameters.pd", clock):
                live = estimate("PAXG", td, bootstrap=0)
            self.assertTrue(live["data_valid"], live["reasons"])

    def test_collector_schema_and_retry(self):
        with tempfile.TemporaryDirectory() as td:
            c = HyperliquidDataCollector(["PAXG"], td, orderbook_depth=2)
            now = pd.Timestamp.now(tz="UTC")
            for depth in (1, 2, 1):
                levels = [[{"px": str(100 + sign * (.01 + i * .01)), "sz": "1"}
                           for i in range(depth)] for sign in (-1, 1)]
                c._on_message(None, json.dumps({"channel": "l2Book", "data": {
                    "coin": "PAXG", "time": int(now.timestamp() * 1000), "levels": levels}}))
            with patch("hyperliquid_data_collector.pq.write_table", side_effect=OSError("disk failure")), \
                    self.assertLogs("hyperliquid_data_collector", level="ERROR"):
                self.assertFalse(c._flush_buffers(force=True))
            self.assertEqual(len(c.buffers[("PAXG", "orderbooks")]), 3)
            self.assertTrue(c._flush_buffers(force=True))
            df = pd.read_parquet(Path(td) / "orderbooks_PAXG.parquet")
            self.assertEqual(len(df), 3)
            self.assertEqual(df.bid_price_1.dtype, float)
            self.assertEqual(len(c.buffers[("PAXG", "orderbooks")]), 0)
            c.fatal = "unresolved capture loss"
            c._health()
            with self.assertRaises(RuntimeError):
                HyperliquidDataCollector(["PAXG"], td)

    def test_stalled_stream_reconnects_and_requires_fresh_data(self):
        with tempfile.TemporaryDirectory() as td, patch("hyperliquid_data_collector.time.time", return_value=1000.) as clock:
            c = HyperliquidDataCollector(["PAXG"], td)
            c.ws = Mock()
            c._on_open(c.ws)
            c.acks = {("PAXG", s) for s in ("trades", "l2Book", "activeAssetCtx")}
            c.last_received = {f"PAXG/{kind}": 1000. for kind in ("orderbooks", "trades", "contexts")}
            clock.return_value = 1040.
            with self.assertLogs("hyperliquid_data_collector", level="WARNING"):
                c._check_connection()
            c.ws.close.assert_called_once()
            self.assertFalse(c.connected)
            c._on_open(c.ws)
            c._health()
            self.assertFalse(json.loads((Path(td) / "health.json").read_text())["symbols"]["PAXG"]["healthy"])
            self.assertFalse(c.last_received)
            c.acks = {("PAXG", s) for s in ("trades", "l2Book", "activeAssetCtx")}
            c.last_received = {f"PAXG/{kind}": 1040. for kind in ("orderbooks", "trades", "contexts")}
            c._health()
            self.assertTrue(json.loads((Path(td) / "health.json").read_text())["symbols"]["PAXG"]["healthy"])
            self.assertIsNone(c.fatal)

    def test_connection_retry_and_interruptible_shutdown(self):
        with tempfile.TemporaryDirectory() as td:
            c = HyperliquidDataCollector(["PAXG"], td)
            sockets = []
            def connect(*args, **callbacks):
                ws = Mock()
                sockets.append(ws)
                def run(**kwargs):
                    callbacks["on_open"](ws)
                    if len(sockets) == 3:
                        c.stop_collection()
                    else:
                        callbacks["on_error"](ws, TimeoutError("simulated connection loss"))
                ws.run_forever.side_effect = run
                return ws
            with patch("hyperliquid_data_collector.websocket.WebSocketApp", side_effect=connect), \
                    patch("hyperliquid_data_collector.websocket.setdefaulttimeout") as timeout, \
                    patch("hyperliquid_data_collector.threading.Thread"), \
                    patch.object(c.stop_event, "wait") as wait, \
                    self.assertLogs("hyperliquid_data_collector", level="WARNING"):
                c.start_collection()
            timeout.assert_called_once_with(10)
            self.assertEqual([call.args[0] for call in wait.call_args_list], [2, 4])
            self.assertTrue(c.stop_event.is_set())
            self.assertTrue(all(ws.send.call_count == 4 for ws in sockets))
            self.assertIsNone(HyperliquidDataCollector(["PAXG"], td).fatal)

    def test_restart_restores_partial_order_deadline_and_trial(self):
        from avellaneda import avellaneda
        from freqtrade.exchange import Exchange
        from freqtrade.persistence import init_db, Order, Trade
        now = pd.Timestamp.now(tz="UTC")
        filled_at = now - pd.Timedelta(minutes=31)
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {
                "AVELLANEDA_STATE": str(Path(td) / "trial.json")}):
            init_db(f"sqlite:///{td}/trades.sqlite")
            trade = Trade(pair="PAXG/USDC:USDC", exchange="hyperliquid", is_open=True,
                          open_rate=99.9, open_date=filled_at.to_pydatetime(), amount=.2,
                          stake_amount=19.98, fee_open=.00015, fee_close=.00045)
            trade.orders = [Order(order_id="paper-restart", ft_order_side="buy", ft_pair=trade.pair,
                                  ft_is_open=True, ft_amount=.5, ft_price=99.9,
                                  status="open", order_type="limit", amount=.5,
                                  filled=.2, remaining=.3, price=99.9, cost=19.98,
                                  order_date=filled_at.to_pydatetime())]
            Trade.session.add(trade)
            Trade.commit()
            first = avellaneda._first_fill(trade, now)
            Trade.session.remove()  # reload from SQLite, with the exchange's in-memory order cache empty
            atomic_json(Path(td) / "trial.json", {"started_at": (now - pd.Timedelta(days=1)).isoformat(),
                                                 "peak_equity": 1002., "stopped": None})
            config = {"dry_run": True, "max_open_trades": 1, "user_data_dir": Path(td)}
            s = avellaneda(config)
            s.dp = SimpleNamespace(current_whitelist=lambda: ["PAXG/USDC:USDC"])
            s.bot_start()
            restored = Trade.get_trades_proxy(is_open=True)[0]
            exchange = SimpleNamespace(_dry_run_open_orders={}, _ft_has={"stop_price_prop": "stopPrice"})
            order = Exchange.fetch_dry_run_order(exchange, "paper-restart")
            self.assertEqual((order["filled"], order["remaining"], order["status"]), (.2, .3, "open"))
            self.assertEqual(s._first_fill(restored, now), first)
            s.params_valid = True
            self.assertEqual(s.custom_exit(restored.pair, restored, now, 100., 0.), "holding_deadline")
            s.state = update_trial(s.state, 991.9, now, True)
            atomic_json(s.state_path, s.state)
            restarted = avellaneda(config)
            restarted.dp = s.dp
            restarted.bot_start()
            self.assertEqual(restarted.state["stopped"], "trial_drawdown")
            self.assertEqual(restarted.state["peak_equity"], 1002.)
            self.assertFalse(restarted.entries_allowed)
            Trade.session.remove()

    def test_trade_identity_and_event_time_loading(self):
        with tempfile.TemporaryDirectory() as td:
            now = pd.Timestamp.now(tz="UTC").timestamp()
            path = Path(td) / "trades.parquet"
            pd.DataFrame({"timestamp": [now, now + 1, now + 1],
                          "exchange_timestamp": [int(now * 1000), int(now * 1000) + 1000, int(now * 1000) + 1000],
                          "symbol": "PAXG", "price": 100., "size": 1., "side": "buy", "trade_id": "same"}).to_parquet(path)
            os.utime(path, (1, 1))  # copied/old mtime must not exclude current event times
            out = load_trades_data(path, pd.Timestamp(now - 1, unit="s", tz="UTC"),
                                   pd.Timestamp(now + 2, unit="s", tz="UTC"))
            self.assertEqual(len(out), 2)

    def test_funding_reconnect_does_not_double_charge(self):
        with tempfile.TemporaryDirectory() as td:
            now = pd.Timestamp.now(tz="UTC").timestamp()
            path = Path(td) / "funding.parquet"
            pd.DataFrame({"timestamp": [now, now + 1], "exchange_timestamp": [int(now * 1000)] * 2,
                          "symbol": "PAXG", "funding_rate": .00001, "mark_price": None}).to_parquet(path)
            self.assertEqual(len(load_funding_data(path)), 1)

    def test_strategy_rejects_bad_update_and_protects_exit(self):
        from avellaneda import avellaneda
        now = pd.Timestamp.now(tz="UTC")
        p, b = parameters(now), live_book(now)
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {
            "AVELLANEDA_PARAMS_DIR": td, "AVELLANEDA_STATE": str(Path(td) / "state.json")}):
            s = avellaneda({"dry_run": True, "max_open_trades": 1, "user_data_dir": Path(td),
                            "dry_run_wallet": 1000, "avellaneda": {"paper_evaluation": True}})
            s.dp = SimpleNamespace(current_whitelist=lambda: [p["pair"]],
                                   market=lambda pair: {"maker": .00015, "taker": .00045},
                                   orderbook=lambda *a, **kw: b)
            with patch("avellaneda.Trade.get_trades_proxy", return_value=[]):
                s.bot_start()
            atomic_json(Path(td) / "avellaneda_parameters_PAXG.json", p)
            s._load_params(now)
            self.assertTrue(s.params_valid)
            s.entries_allowed = True
            bad = copy.deepcopy(p)
            bad["intensity"]["bid"]["k"] = 0.
            atomic_json(Path(td) / "avellaneda_parameters_PAXG.json", bad)
            self.assertFalse(s.confirm_trade_entry(p["pair"], "limit", .5, 99.9, "GTC", now))
            s.protective_reason = "holding_deadline"
            price = s.custom_exit_price(p["pair"], SimpleNamespace(amount=.5), now, 100.01, -.01, "holding_deadline")
            self.assertEqual(price, 99.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
