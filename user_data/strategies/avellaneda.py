"""Single-position, paper-only constrained market maker; all entries fail closed."""
import json
import logging
import math
import os
from pathlib import Path
import sys

import pandas as pd
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from quote_model import HORIZON, POSITION_STOP, STAKE, liquidation, quote_decision, update_trial, validate_book, validate_params, warm_quote_kernel
from utils import atomic_json, get_tick_size, utc

log = logging.getLogger(__name__)


class avellaneda(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "15m"
    process_only_new_candles = False
    startup_candle_count = 0
    position_adjustment_enable = False
    minimal_roi = {}
    stoploss = -POSITION_STOP
    trailing_stop = False
    order_types = {"entry": "limit", "exit": "limit", "stoploss": "market",
                   "emergency_exit": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def bot_start(self, **kwargs):
        if not self.config.get("dry_run"):
            raise OperationalException("This research implementation is paper-only")
        pairs = self.dp.current_whitelist()
        if len(pairs) != 1 or self.config.get("max_open_trades") != 1:
            raise OperationalException("Exactly one pair and one open position are required")
        if self.config.get("fee") is not None:
            raise OperationalException("Remove the fee override: maker and taker costs must be distinct")
        warm_quote_kernel()
        self.pair = pairs[0]
        options = self.config.get("avellaneda", {})
        self.paper_evaluation = options.get("paper_evaluation", False)
        if type(self.paper_evaluation) is not bool:
            raise OperationalException("paper_evaluation must be boolean")
        self.params_dir = Path(os.getenv("AVELLANEDA_PARAMS_DIR", str(Path(__file__).resolve().parents[2] / "scripts")))
        self.state_path = Path(os.getenv("AVELLANEDA_STATE", str(Path(self.config["user_data_dir"]) / "avellaneda_trial.json")))
        self.params, self.entries_allowed, self.decision = None, False, None
        self.params_valid, self.failure, self.book = False, "awaiting_parameters", None
        self.protective_reason = None
        self.last_loop = None
        try:
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
                if set(("started_at", "peak_equity", "stopped")) - self.state.keys():
                    raise ValueError("Incomplete trial state")
                if self.state["started_at"] is not None:
                    utc(self.state["started_at"])
                    if not math.isfinite(self.state["peak_equity"]):
                        raise ValueError("Invalid high-water mark")
            elif Trade.get_trades_proxy():
                raise ValueError("Existing trades without trial state; refusing to reset losses")
            else:
                self.state = {"started_at": None, "peak_equity": None, "stopped": None}
                atomic_json(self.state_path, self.state)
        except Exception as exc:
            raise OperationalException(f"Cannot restore trial state: {exc}") from exc

    def _load_params(self, now):
        self.params_valid = False
        try:
            path = self.params_dir / f"avellaneda_parameters_{self.pair.split('/')[0]}.json"
            candidate = json.loads(path.read_text(encoding="utf-8"))
            validate_params(candidate, self.pair, now)
            market = self.dp.market(self.pair)
            if market:
                for fee in ("maker", "taker"):
                    if not math.isclose(float(market[fee]), candidate["market"][fee + "_fee"], abs_tol=1e-10):
                        raise ValueError("Fee metadata differs from Freqtrade")
            self.params = candidate
            self.params_valid = True
            self.failure = None
        except Exception as exc:
            self.failure = str(exc)

    def _get_book(self, pair, now):
        book = dict(self.dp.orderbook(pair, maximum=20))
        book["received_at"] = pd.Timestamp.now(tz="UTC").timestamp()
        validate_book(book, pd.Timestamp.now(tz="UTC"))
        return book

    @staticmethod
    def _first_fill(trade, now):
        saved = trade.get_custom_data(key="first_fill_at")
        if saved is None:
            fills = [o for o in trade.orders if o.ft_order_side == trade.entry_side and (o.filled or 0) > 0]
            dates = [o.order_filled_utc or o.order_date_utc for o in fills]
            at = min(dates) if dates else now
            saved = utc(at).isoformat()
            trade.set_custom_data(key="first_fill_at", value=saved)
        return utc(saved)

    def _position_profit(self, trade):
        market = self.dp.market(trade.pair)
        fee = float(market["taker"])
        proceeds, limit = liquidation(self.book, trade.amount, fee)
        # Native calculation includes already-paid entry fees and signed funding.
        rate = proceeds / trade.amount / (1 - fee)
        profit = trade.calculate_profit(rate).profit_abs - trade.amount * rate * (fee - trade.fee_close)
        return profit + float(trade.realized_profit or 0), limit

    def bot_loop_start(self, current_time, **kwargs):
        self.entries_allowed = False
        self.decision = None
        self.protective_reason = None
        now = utc(current_time)
        try:
            self._load_params(now)
            self.book = self._get_book(self.pair, now)
            now = pd.Timestamp.now(tz="UTC")
            positions = [t for t in Trade.get_trades_proxy(is_open=True) if t.amount > 0]
            equity = float(self.config.get("dry_run_wallet", 1000)) + Trade.get_total_closed_profit()
            for trade in positions:
                profit, _ = self._position_profit(trade)
                equity += profit
                first = self._first_fill(trade, now)
                notional = trade.max_stake_amount or trade.open_rate * trade.amount
                if profit <= -POSITION_STOP * notional:
                    self.protective_reason = "position_stop"
                if (now - first).total_seconds() >= HORIZON:
                    self.protective_reason = "holding_deadline"
            eligible = self.params_valid and (self.paper_evaluation or self.params["trade_enabled"])
            self.state = update_trial(self.state, equity, now, eligible)
            if self.state["stopped"]:
                self.protective_reason = self.state["stopped"]
            if positions and not self.params_valid:
                self.protective_reason = "invalid_parameters"
            if self.params_valid and not self.protective_reason:
                q = positions[0].amount if positions else 0.
                remaining = HORIZON
                if positions:
                    remaining -= (now - self._first_fill(positions[0], now)).total_seconds()
                self.decision = quote_decision(self.params, self.book, now, q, remaining)
                self.entries_allowed = bool(eligible and not positions and self.decision["action"] == "buy")
            self.state.update(last_loop=now.isoformat(), status="stopped" if self.state["stopped"] else
                              ("ready" if eligible else "waiting_for_valid_data"),
                              reason=self.protective_reason or self.failure,
                              decision=self.decision, equity=equity)
            atomic_json(self.state_path, self.state)
            self.last_loop = now
            log.info("MM state: %s", json.dumps(self.state, allow_nan=False))
        except Exception as exc:
            self.entries_allowed = False
            self.params_valid = False
            self.failure = str(exc)
            self.protective_reason = "invalid_runtime_state"
            log.exception("Entries blocked; exit management remains active")

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe["enter_long"] = int(bool(getattr(self, "entries_allowed", False)))
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_long"] = 0
        return dataframe

    def custom_entry_price(self, pair, current_time, proposed_rate, entry_tag=None, side="long", **kwargs):
        if not self.entries_allowed or side != "long":
            self.entries_allowed = False
            return proposed_rate  # confirm_trade_entry vetoes the framework fallback
        return self.decision["price"]

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        if not self.entries_allowed or not self.decision:
            return 0.
        stake = self.decision["quantity"] * current_rate
        return stake if (min_stake or 0) <= stake <= min(STAKE, max_stake) + 1e-8 else 0.

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time,
                            entry_tag=None, side="long", **kwargs):
        try:
            now = utc(current_time)
            if not self.entries_allowed or side != "long" or self.state["stopped"] or not self.decision:
                return False
            # Re-read at the final boundary; a failed update cannot leave a cached buy enabled.
            self._load_params(now)
            if not self.params_valid or not (self.paper_evaluation or self.params["trade_enabled"]):
                return False
            if self.last_loop is None or (now - self.last_loop).total_seconds() > 20:
                return False
            book = self._get_book(pair, now)
            tick = get_tick_size(rate, self.params["market"]["sz_decimals"])
            return bool(math.isfinite(rate) and math.isfinite(amount) and amount > 0 and
                        abs(rate - self.decision["price"]) < tick * .51 and
                        abs(amount - self.decision["quantity"]) < 10 ** -self.params["market"]["sz_decimals"] * .51 and
                        amount * rate <= STAKE + 1e-8 and rate < book["asks"][0][0])
        except Exception:
            self.entries_allowed = False
            log.exception("Entry confirmation failed closed")
            return False

    def check_entry_timeout(self, pair, trade, order, current_time, **kwargs):
        # No automatic replacements bypassing confirmation: cancel then enter anew.
        return bool(not self.entries_allowed or (order.filled or 0) > 0 or
                    (utc(current_time) - utc(order.order_date_utc)).total_seconds() >= 4)

    def check_exit_timeout(self, pair, trade, order, current_time, **kwargs):
        return bool(self.protective_reason or not self.decision or self.decision["action"] == "wait" or
                    (utc(current_time) - utc(order.order_date_utc)).total_seconds() >= 4)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if self.protective_reason:
            return self.protective_reason
        if not self.params_valid:
            return "invalid_parameters"
        if (utc(current_time) - self._first_fill(trade, current_time)).total_seconds() >= HORIZON:
            return "holding_deadline"
        if self.decision and self.decision["action"] in ("sell", "liquidate"):
            return "model_exit" if self.decision["action"] == "sell" else "model_liquidation"
        return None

    def custom_exit_price(self, pair, trade, current_time, proposed_rate, current_profit, exit_tag=None, **kwargs):
        try:
            book = self._get_book(pair, current_time)
            if exit_tag == "model_exit" and self.params_valid and not self.protective_reason and self.decision:
                return self.decision["price"]
            # Protective exits never use the passive model ask.
            fee = float(self.dp.market(pair)["taker"])
            try:
                return liquidation(book, trade.amount, fee)[1]
            except ValueError:
                return float(book["bids"][-1][0])  # execute available depth; keep managing the residual
        except Exception:
            self.entries_allowed = False
            log.exception("Exit book unavailable; use Freqtrade's current exit rate")
            return proposed_rate

    def leverage(self, **kwargs):
        return 1.
