"""Public Hyperliquid capture with typed, atomically published parquet batches."""
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from collections import defaultdict, deque

import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import websocket

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils import atomic_json

log = logging.getLogger(__name__)
KINDS = ("orderbooks", "trades", "prices", "contexts", "funding")


def schema(kind, depth):
    fields = [("timestamp", pa.float64()), ("exchange_timestamp", pa.int64()), ("symbol", pa.string())]
    if kind == "orderbooks":
        fields += [(f"{side}_{name}_{i}", pa.float64())
                   for i in range(depth) for side in ("bid", "ask") for name in ("price", "size")]
    elif kind in ("prices", "trades"):
        fields += [("price", pa.float64()), ("size", pa.float64()), ("side", pa.string())]
        if kind == "trades":
            fields += [("trade_id", pa.string())]
    else:
        fields += [("funding_rate", pa.float64()), ("mark_price", pa.float64())]
    return pa.schema(fields)


class HyperliquidDataCollector:
    def __init__(self, symbols, output_dir="HL_data", orderbook_depth=20, rotation_interval=300):
        self.symbols = [s.upper() for s in symbols]
        if not self.symbols or not 1 <= orderbook_depth <= 20:
            raise ValueError("At least one symbol and 1..20 book levels required")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        previous_health = self.output_dir / "health.json"
        if previous_health.exists() and json.loads(previous_health.read_text()).get("fatal"):
            raise RuntimeError("Unresolved capture failure in health.json; inspect the data before restarting")
        self.depth, self.rotation_interval = orderbook_depth, rotation_interval
        self.buffers = {(s, k): deque() for s in self.symbols for k in KINDS}
        self.schemas = {k: schema(k, self.depth) for k in KINDS}
        self.last_publish = defaultdict(time.time)
        self.last_received, self.acks = {}, set()
        self.received, self.written = defaultdict(int), defaultdict(int)
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.running, self.connected, self.writer_healthy = False, False, True
        self.connected_at = 0.
        self.fatal, self.ws = None, None
        self.metadata_time = 0.
        self.metadata_attempt = 0.
        self.metadata_error = None
        self.last_funding = {}
        self.exchange = None
        self.rejected = 0

    def _append(self, symbol, kind, row):
        if symbol not in self.symbols:
            return
        key = (symbol, kind)
        with self.lock:
            if len(self.buffers[key]) >= 100_000:
                self.fatal = f"Unpublished buffer overflow: {symbol}/{kind}"
                raise BufferError(self.fatal)
            self.buffers[key].append(row)
            self.received[f"{symbol}/{kind}"] += 1
            self.last_received[f"{symbol}/{kind}"] = time.time()

    def _on_open(self, ws):
        self.connected = True
        self.connected_at = time.time()
        self.acks.clear()
        self.last_received.clear()  # freshness must come from this connection
        for coin in self.symbols:
            for stream in ("trades", "l2Book", "bbo", "activeAssetCtx"):
                sub = {"type": stream, "coin": coin}
                if stream == "l2Book":
                    sub["fast"] = True  # measured ~0.55 s / five levels; default is ~5 s / twenty
                ws.send(json.dumps({"method": "subscribe", "subscription": sub}))

    def _check_connection(self):
        now = time.time()
        if not self.connected or now - self.connected_at < 30:
            return
        missing = any((coin, stream) not in self.acks for coin in self.symbols
                      for stream in ("trades", "l2Book", "activeAssetCtx"))
        stale = any(now - self.last_received.get(f"{coin}/{kind}", 0) > 30
                    for coin in self.symbols for kind in ("orderbooks", "contexts"))
        if missing or stale:
            log.warning("Market stream stalled; reconnecting and resubscribing")
            self.connected = False
            self.ws.close()

    def _on_message(self, ws, message):
        try:
            packet = json.loads(message)
            channel, data = packet.get("channel"), packet.get("data")
            if channel == "subscriptionResponse":
                sub = data["subscription"]
                self.acks.add((sub["coin"], sub["type"]))
                return
            now = time.time()
            if channel == "trades":
                for trade in data:
                    if trade["side"] not in ("A", "B") or trade.get("tid") is None:
                        raise ValueError("Invalid trade side/id")
                    price, size = float(trade["px"]), float(trade["sz"])
                    if not (0 < price < float("inf") and 0 < size < float("inf")):
                        raise ValueError("Invalid trade price/size")
                    self._append(trade["coin"], "trades",
                                 {"timestamp": now, "exchange_timestamp": int(trade["time"]),
                                  "symbol": trade["coin"], "price": price, "size": size,
                                  "side": "sell" if trade["side"] == "A" else "buy",
                                  "trade_id": str(trade["tid"])})
            elif channel == "l2Book":
                row = {"timestamp": now, "exchange_timestamp": int(data["time"]), "symbol": data["coin"]}
                for side, levels in zip(("bid", "ask"), data["levels"]):
                    for i in range(self.depth):
                        row[f"{side}_price_{i}"] = float(levels[i]["px"]) if i < len(levels) else None
                        row[f"{side}_size_{i}"] = float(levels[i]["sz"]) if i < len(levels) else None
                self._append(data["coin"], "orderbooks", row)
            elif channel == "bbo":
                for side, level in zip(("bid", "ask"), data["bbo"]):
                    if level is not None:
                        self._append(data["coin"], "prices",
                                     {"timestamp": now, "exchange_timestamp": int(data["time"]),
                                      "symbol": data["coin"], "price": float(level["px"]),
                                      "size": float(level["sz"]), "side": side})
            elif channel == "activeAssetCtx":
                ctx = data["ctx"]
                self._append(data["coin"], "contexts",
                             {"timestamp": now, "exchange_timestamp": int(now * 1000), "symbol": data["coin"],
                              "funding_rate": float(ctx["funding"]), "mark_price": float(ctx["markPx"])})
        except Exception as exc:
            self.rejected += 1
            log.exception("Rejected market message: %s", exc)
            # Invalid capture is visible to readers instead of being silently treated as no trading.
            self.fatal = str(exc)
            if ws is not None:
                ws.close()

    def _flush_buffers(self, force=False):
        with self.write_lock:
            for (symbol, kind), rows in self.buffers.items():
                key = (symbol, kind)
                with self.lock:
                    if not rows or (not force and len(rows) < 10_000 and
                                    time.time() - self.last_publish[key] < self.rotation_interval):
                        continue
                    batch = list(rows)
                folder = self.output_dir / f"{kind}_{symbol}.parquet"
                folder.mkdir(exist_ok=True)
                path = folder / f"part_{time.time_ns()}.parquet"
                tmp = path.with_suffix(".tmp")
                try:
                    table = pa.Table.from_pylist(batch, schema=self.schemas[kind])
                    pq.write_table(table, tmp, compression="zstd")
                    with tmp.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(tmp, path)
                    with self.lock:
                        for _ in batch:
                            rows.popleft()
                        self.written[f"{symbol}/{kind}"] += len(batch)
                    self.last_publish[key] = time.time()
                except Exception:
                    self.writer_healthy = False
                    log.exception("Publication failed; %s/%s batch retained", symbol, kind)
                    return False
            self.writer_healthy = True
        return True

    def _metadata(self):
        if self.exchange is None:
            self.exchange = ccxt.hyperliquid({"enableRateLimit": True, "timeout": 10000})
        markets = self.exchange.load_markets(reload=True)
        now = time.time()
        for coin in self.symbols:
            if not self.running:
                return
            pair = f"{coin}/USDC:USDC"
            market = markets[pair]
            size_step = float(market["precision"]["amount"])
            decimals = int(round(-math.log10(size_step)))
            limits = market["limits"]
            metadata = {"symbol": pair, "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                         "sz_decimals": decimals, "maker_fee": float(market["maker"]),
                         "taker_fee": float(market["taker"]),
                         "min_amount": float(limits["amount"].get("min") or size_step),
                         "min_notional": float(limits["cost"].get("min") or 10),
                         "fee_source": "ccxt_public", "ccxt": ccxt.__version__}
            atomic_json(self.output_dir / f"market_{coin}.json", metadata)
            atomic_json(self.output_dir / f"market_{coin}" / f"{int(now)}.json", metadata)
            # Settled history is distinct from the predictive funding context stream.
            history = self.exchange.fetch_funding_rate_history(pair, since=int((now - 86400) * 1000))
            for item in sorted(history, key=lambda h: h["timestamp"]):
                stamp = int(item["timestamp"])
                if stamp > self.last_funding.get(coin, 0):
                    self._append(coin, "funding", {"symbol": coin, "timestamp": now,
                                 "exchange_timestamp": stamp, "funding_rate": float(item["fundingRate"]),
                                 "mark_price": None})
                    self.last_funding[coin] = stamp
        self.metadata_time = now

    def _health(self):
        now = time.time()
        symbols = {}
        for coin in self.symbols:
            acknowledgements = all((coin, s) in self.acks for s in ("trades", "l2Book", "activeAssetCtx"))
            fresh_book = now - self.last_received.get(f"{coin}/orderbooks", 0) <= 10
            recent_trades = now - self.last_received.get(f"{coin}/trades", 0) <= 600
            fresh_context = now - self.last_received.get(f"{coin}/contexts", 0) <= 10
            symbols[coin] = {"subscriptions_confirmed": acknowledgements,
                             "book_fresh": fresh_book,
                             "trades_recent": recent_trades, "context_fresh": fresh_context,
                             "last_trade_at": self.last_received.get(f"{coin}/trades"),
                             "healthy": bool(self.connected and acknowledgements and fresh_book and recent_trades and fresh_context and
                                             self.writer_healthy and not self.fatal and not self.metadata_error)}
        atomic_json(self.output_dir / "health.json",
                    {"timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "symbols": symbols,
                     "received": dict(self.received), "written": dict(self.written),
                     "pending": {f"{s}/{k}": len(v) for (s, k), v in self.buffers.items()},
                     "writer_healthy": self.writer_healthy, "rejected": self.rejected, "fatal": self.fatal,
                     "metadata_error": self.metadata_error})

    def _maintenance(self):
        while self.running:
            try:
                self._check_connection()
                self._flush_buffers()
                if time.time() - self.metadata_time >= 3600 and time.time() - self.metadata_attempt >= 60:
                    self.metadata_attempt = time.time()
                    try:
                        self._metadata()
                        self.metadata_error = None
                    except Exception as exc:
                        self.metadata_error = str(exc)
                        log.exception("Metadata refresh failed; retrying in one minute")
                self._health()
            except Exception:
                self.writer_healthy = False
                log.exception("Collector maintenance failed")
            if self.fatal:
                self.stop_collection()
            self.stop_event.wait(5)

    def stop_collection(self, *args):
        self.running = False
        self.stop_event.set()
        if self.ws:
            self.ws.close()

    def start_collection(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        signal.signal(signal.SIGTERM, self.stop_collection)
        signal.signal(signal.SIGINT, self.stop_collection)
        websocket.setdefaulttimeout(10)
        self.stop_event.clear()
        self.running = True
        retry_delay = 2
        worker = threading.Thread(target=self._maintenance, daemon=False)
        worker.start()
        try:
            while self.running:
                attempt = time.monotonic()
                self.ws = websocket.WebSocketApp("wss://api.hyperliquid.xyz/ws",
                    on_open=self._on_open, on_message=self._on_message,
                    on_error=lambda ws, error: log.warning("WebSocket: %s", error))
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
                self.connected = False
                self.acks.clear()
                if self.running:
                    if time.monotonic() - attempt >= 60:
                        retry_delay = 2
                    log.warning("Reconnecting in %s seconds", retry_delay)
                    self.stop_event.wait(retry_delay)
                    retry_delay = min(30, retry_delay * 2)
        finally:
            self.running = False
            self.stop_event.set()
            worker.join(timeout=60)
            self.connected = False
            self._flush_buffers(force=True)
            self._health()
        if self.fatal:
            raise RuntimeError(self.fatal)
