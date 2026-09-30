# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
import sys
from pandas import DataFrame
from functools import reduce
import json
import logging
from pathlib import Path
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade, Order
from freqtrade.exceptions import OperationalException
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
# --------------------------------
# Add your lib to import here
import math
from typing import Optional, Tuple
from dataclasses import dataclass
from pair_loader import pair_to_ticker


logger = logging.getLogger(__name__)

# Setup dedicated logger for market making values
mm_logger = logging.getLogger('market_making_values')
mm_logger.setLevel(logging.INFO)
if not mm_logger.handlers:  # the module is imported again on /reload_config
    mm_handler = RotatingFileHandler(Path(__file__).parent / 'log_ave_mm.log', maxBytes=10_000_000, backupCount=3)
    mm_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    mm_logger.addHandler(mm_handler)
mm_logger.propagate = False

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None


def calculate_optimal_spreads(mid_price, sigma, k_bid, k_ask, gamma, time_remaining, q_inventory_exposure, fee):
    """Compute reservation price and bid/ask quotes, logging the inputs/outputs."""
    sigma_abs = sigma * mid_price

    reservation_decay = gamma * sigma_abs**2.0 * time_remaining
    risk_term = 0.5 * reservation_decay

    half_spread_bid = risk_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_bid))
    half_spread_ask = risk_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_ask))

    r = mid_price - q_inventory_exposure * reservation_decay

    r_b = r - half_spread_bid - mid_price * fee
    r_a = r + half_spread_ask + mid_price * fee

    delta_a_rel = r_a - mid_price
    delta_b_rel = mid_price - r_b

    delta_a_percent = (delta_a_rel / mid_price) * 100.0
    delta_b_percent = (delta_b_rel / mid_price) * 100.0

    mm_logger.info("=" * 65)
    mm_logger.info("AVELLANEDA-STOIKOV MODEL PARAMETERS")
    mm_logger.info("=" * 65)
    mm_logger.info(f"Time Remaining Fraction: {time_remaining:>12.4f}")
    mm_logger.info(f"Inventory Exposure:      {q_inventory_exposure:>12.4f}")
    mm_logger.info(f"Sigma (Volatility):      {sigma:>12.6f}")
    mm_logger.info(f"K Bid:                   {k_bid:>12.6f}")
    mm_logger.info(f"K Ask:                   {k_ask:>12.6f}")
    mm_logger.info(f"Maker fee:               {fee:>12.6f}")
    mm_logger.info(f"Gamma:                   {gamma:>12.6f}")
    mm_logger.info(f"Mid-Price:               {mid_price:>12.4f}")
    mm_logger.info(f"Reservation Price:       {r:>12.4f}")
    mm_logger.info(f"Buy Spread (% of mid):   {delta_b_percent:>12.4f}%")
    mm_logger.info(f"Sell Spread (% of mid):  {delta_a_percent:>12.4f}%")
    mm_logger.info(f"Buy Limit Price:         {r_b:>12.4f}")
    mm_logger.info(f"Sell Limit Price:        {r_a:>12.4f}")
    mm_logger.info("=" * 65)

    return r_b, r_a


def _fmt_optional(value: float | None, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

#---------------------------------------------------------- LOAD CONFIG ----------------------------------------------------------

def get_params_directory():
    """
    Get the directory containing parameter JSON files.
    Uses environment variable AVELLANEDA_PARAMS_DIR if set, otherwise searches for scripts/ directory.
    Works consistently whether running locally, in Docker, or in a container.

    Returns:
        Path: Directory where parameter files are located
    """
    import os

    # First check environment variable
    env_path = os.getenv('AVELLANEDA_PARAMS_DIR')
    if env_path:
        params_dir = Path(env_path).resolve()
        logger.debug(f"Using params directory from AVELLANEDA_PARAMS_DIR: {params_dir}")
        return params_dir

    # Fall back to scripts directory relative to project root
    try:
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
    except NameError:  # e.g., interactive
        current_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()

    # Search for scripts directory
    search_paths = [
        current_dir / '../../scripts',  # From user_data/strategies/
        current_dir / '../scripts',
        current_dir / 'scripts',
        current_dir.parent.parent / 'scripts',
    ]

    for path in search_paths:
        resolved = path.resolve()
        if resolved.exists() and resolved.is_dir():
            logger.debug(f"Using params directory: {resolved}")
            return resolved

    # If scripts/ not found, use current directory as fallback
    logger.warning(f"scripts/ directory not found, using current directory: {current_dir}")
    return current_dir


def load_configs(pair: str) -> dict | None:
    """Parameters written by scripts/calculate_avellaneda_parameters.py for `pair`; None if missing or unreadable."""
    params_file = get_params_directory() / f"avellaneda_parameters_{pair_to_ticker(pair)}.json"
    try:
        return json.loads(params_file.read_text(encoding='utf-8'))
    except FileNotFoundError:
        logger.debug(f"No parameter file yet at {params_file}")
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error reading {params_file}: {e}")
    return None


def log_parameters_summary(params: dict) -> None:
    md, op = params.get('market_data', {}), params.get('optimal_parameters', {})
    logger.info(
        "New parameters | computed=%s | data_end=%s | trade_enabled=%s | gamma=%s | sigma=%s | k_bid=%s | k_ask=%s "
        "| T_h=%s | fee=%s",
        params.get('timestamp'), params.get('data_end'), params.get('trade_enabled'),
        _fmt_optional(op.get('gamma')), _fmt_optional(md.get('sigma')), _fmt_optional(md.get('k_bid')),
        _fmt_optional(md.get('k_ask')), _fmt_optional(op.get('time_horizon_hours'), 4), params.get('maker_fee'),
    )


class avellaneda(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = False
    position_adjustment_enable: bool = False
    max_entry_position_adjustment = 0
    startup_candle_count: int = 0

    minimal_roi = {
        "0": -1
    }

    params_MM = None
    gamma = None
    k_bid = None
    k_ask = None
    sigma = None
    time_horizon_hours = None

    maker_fee = 0.0002                        # overwritten by "maker_fee" from the parameter file
    entries_allowed: Optional[bool] = None    # None until the first parameter load
    max_param_age = timedelta(hours=1)        # no new entries on parameters from older data

    stoploss = -0.85

    trailing_stop = False

    timeframe = '15m'

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        "emergency_exit": "limit",
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pairs = self.dp.current_whitelist()
        if len(pairs) != 1:
            raise OperationalException(f"avellaneda trades exactly one pair; whitelist has {len(pairs)}: {pairs}")
        self._load_params()

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Parameters are recomputed by the hl-params service (scripts/calculate_avellaneda_parameters.py)."""
        self._load_params()

    def _load_params(self) -> None:
        """
        Reload the calculator's JSON; a missing or unreadable file keeps the last good values so open trades can
        still exit. New entries only if the calculator found an edge on data younger than max_param_age.
        """
        params = load_configs(self.dp.current_whitelist()[0])
        if params and params.get('timestamp') != (self.params_MM or {}).get('timestamp'):
            try:
                md, op = params['market_data'], params['optimal_parameters']
                new = op['gamma'], md['sigma'], md['k_bid'], md['k_ask'], op['time_horizon_hours']
            except KeyError as e:
                logger.error(f"Parameter file lacks {e}; keeping previous parameters")
            else:  # all-or-nothing update
                self.gamma, self.sigma, self.k_bid, self.k_ask, self.time_horizon_hours = new
                self.maker_fee, self.params_MM = params.get('maker_fee', self.maker_fee), params
                log_parameters_summary(params)

        p = self.params_MM or {}
        age = pd.Timestamp.now(tz='UTC') - pd.Timestamp(p['data_end']) if p.get('data_end') else None
        allowed = bool(p.get('trade_enabled')) and age is not None and age < self.max_param_age
        if allowed != self.entries_allowed:
            why = f"trade_enabled={p.get('trade_enabled')}, data age={age}" if p else "no parameter file yet"
            logger.info(f"New entries {'enabled' if allowed else 'blocked'} ({why})")
            self.entries_allowed = allowed

    def informative_pairs(self):
        """
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter only while fresh parameters show an edge (see _load_params); exits are never blocked."""
        dataframe.loc[:, 'enter_long'] = int(bool(self.entries_allowed))
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
    
    def get_mid_price(self, pair: str, fallback_rate: float) -> float:
        """
        Get effective mid price from orderbook based on $1000 depth.
        Fallback to best mid or provided rate if orderbook unavailable or insufficient.
        """
        # Request deeper orderbook to find $1000 depth
        orderbook = self.dp.orderbook(pair, maximum=50)
        
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return fallback_rate
            
        THRESHOLD = 1000.0
        
        # Naive Mid Calculation (Top of Book)
        best_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        best_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
        naive_mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        
        # Calculate Effective Bid
        effective_bid = None
        cum_val = 0.0
        # Bids are sorted high to low
        for price, amount in orderbook['bids']:
            cum_val += price * amount
            if cum_val >= THRESHOLD:
                effective_bid = price
                break
        
        # Fallback to best bid if threshold not reached
        if effective_bid is None and len(orderbook['bids']) > 0:
            effective_bid = orderbook['bids'][0][0]
            
        # Calculate Effective Ask
        effective_ask = None
        cum_val = 0.0
        # Asks are sorted low to high
        for price, amount in orderbook['asks']:
            cum_val += price * amount
            if cum_val >= THRESHOLD:
                effective_ask = price
                break
                
        # Fallback to best ask if threshold not reached
        if effective_ask is None and len(orderbook['asks']) > 0:
            effective_ask = orderbook['asks'][0][0]
            
        if effective_bid is not None and effective_ask is not None:
            effective_mid = (effective_bid + effective_ask) / 2
            
            # Log comparison
            if naive_mid:
                diff_pct = abs(effective_mid - naive_mid) / naive_mid * 100
                logger.info(f"Price Check | Naive Mid: {naive_mid:.4f} | Effective Mid: {effective_mid:.4f} | Diff: {diff_pct:.4f}%")
            
            return effective_mid
        else:
            return fallback_rate
     
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:

        if self.sigma is None or self.gamma is None or self.params_MM is None:
            return None

        if side!="long":
            return None
        
        mid_price = self.get_mid_price(pair, proposed_rate)
        # Inventory term off (q = 0): long-only, one position; scripts/backtest.py simulates exactly this
        r_buy, r_sell = calculate_optimal_spreads(mid_price, self.sigma, self.k_bid, self.k_ask, self.gamma,
                                                  self.time_horizon_hours / 24.0, 0.0, self.maker_fee)

        return r_buy

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:
        
        if self.sigma is None or self.gamma is None or self.params_MM is None:
            return None

        if trade.is_short:
            return None

        mid_price = self.get_mid_price(pair, proposed_rate)
        # Inventory term off (q = 0): long-only, one position; scripts/backtest.py simulates exactly this
        r_buy, r_sell = calculate_optimal_spreads(mid_price, self.sigma, self.k_bid, self.k_ask, self.gamma,
                                                  self.time_horizon_hours / 24.0, 0.0, self.maker_fee)

        return r_sell

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        return "always_exit"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        lev = 1
        logger.info(f"Using leverage: {lev}. Should not be changed.")
        return lev

    # @property
    # def protections(self):
    #     return [
    #         {
    #             "method": "MaxDrawdown",
    #             "lookback_period": 10080,  # 1 week
    #             "trade_limit": 0,  # Evaluate all trades since the bot started
    #             "stop_duration_candles": 10000000,  # Stop trading indefinitely
    #             "max_allowed_drawdown": 0.05  # Maximum drawdown of 5% before stopping
    #         },
    #     ]
