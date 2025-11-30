# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
import sys
import threading
from run_avellaneda_param_calculation import run_avellaneda_param_calculation
from pandas import DataFrame
from functools import reduce
import json
import logging
from pathlib import Path
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade, Order
from datetime import datetime, timezone
# --------------------------------
# Add your lib to import here
import math
from typing import Optional, Tuple
from dataclasses import dataclass
from pair_loader import get_active_pair, pair_to_ticker


logger = logging.getLogger(__name__)

# Setup dedicated logger for market making values
mm_logger = logging.getLogger('market_making_values')
mm_logger.setLevel(logging.INFO)
log_file_path = Path(__file__).parent / 'log_ave_mm.txt'
mm_handler = logging.FileHandler(log_file_path)
mm_formatter = logging.Formatter('%(asctime)s - %(message)s')
mm_handler.setFormatter(mm_formatter)
mm_logger.addHandler(mm_handler)
mm_logger.propagate = False

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

def calculate_optimal_spreads(mid_price,sigma,k_bid,k_ask,gamma,time_remaining,q_inventory_exposure, fee):
    # Convert percentage volatility to absolute volatility (in $)
    sigma_abs = sigma * mid_price

    # Calculate reservation price decay and risk term
    reservation_decay = gamma * sigma_abs**2.0 * time_remaining
    risk_term = 0.5 * reservation_decay

    # Calculate asymmetric half-spreads
    # Note: If k is very small, log term dominates.
    half_spread_bid = risk_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_bid))
    half_spread_ask = risk_term + (1.0 / gamma) * np.log(1.0 + (gamma / k_ask))

    # Calculate reservation price (our "fair value" given inventory and time remaining)
    r = mid_price - q_inventory_exposure * reservation_decay

    # Calculate limit order prices
    # Quotes are centered around reservation price r, but with asymmetric spreads
    r_b = r - half_spread_bid - mid_price*fee
    r_a = r + half_spread_ask + mid_price*fee

    # Calculate deltas relative to mid-price for logging
    delta_a_rel = r_a - mid_price
    delta_b_rel = mid_price - r_b

    # Calculate relative percentages
    delta_a_percent = (delta_a_rel / mid_price) * 100.0
    delta_b_percent = (delta_b_rel / mid_price) * 100.0

    mm_logger.info("=" * 65)
    mm_logger.info("📊 AVELLANEDA-STOIKOV MODEL PARAMETERS")
    mm_logger.info("=" * 65)
    mm_logger.info(f"│ Time Remaining Fraction    │ {time_remaining:>12.4f}          │")
    mm_logger.info(f"│ Inventory Exposure         │ {q_inventory_exposure:>12.4f}          │")
    mm_logger.info(f"│ Sigma (Volatility)         │ {sigma:>12.4f}          │")
    mm_logger.info(f"│ K Bid                      │ {k_bid:>12.4f}          │")
    mm_logger.info(f"│ K Ask                      │ {k_ask:>12.4f}          │")
    mm_logger.info(f"│ maker fee                  │ {fee:>12.5f}          │")
    mm_logger.info(f"│ Gamma                      │ {gamma:>12.4f}          │")
    mm_logger.info(f"│ Mid-Price                  │ {mid_price:>12.4f}          │")
    mm_logger.info(f"│ Reservation Price          │ {r:>12.4f}          │")
    mm_logger.info("├" + "─" * 63 + "┤")
    mm_logger.info(f"│ Buy Spread (% of mid)      │ {delta_b_percent:>12.4f}%         │")
    mm_logger.info(f"│ Sell Spread (% of mid)     │ {delta_a_percent:>12.4f}%         │")
    mm_logger.info(f"│ Buy Limit Price            │ {r_b:>12.4f}          │")
    mm_logger.info(f"│ Sell Limit Price           │ {r_a:>12.4f}          │")
    mm_logger.info("=" * 65)

    return r_b, r_a

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
        logger.info(f"Using params directory from AVELLANEDA_PARAMS_DIR: {params_dir}")
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
            logger.info(f"Using params directory: {resolved}")
            return resolved

    # If scripts/ not found, use current directory as fallback
    logger.warning(f"scripts/ directory not found, using current directory: {current_dir}")
    return current_dir


def find_upwards(filename: str, start: Path, max_up: int = 10) -> Path:
    p = start.resolve()
    for _ in range(max_up + 1):
        candidate = p / filename
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError(f"Could not find {filename} from {start}")


def load_configs(start_dir: Path | None = None, max_up: int = 10):
    """
    Load Avellaneda parameters from JSON file, searching in multiple locations.

    Args:
        start_dir: Starting directory for search (defaults to current file's directory)
        max_up: Maximum levels to search upwards

    Returns:
        dict: Loaded parameters from JSON file

    Raises:
        FileNotFoundError: If the parameters file cannot be found
    """
    params_dir = get_params_directory()

    try:
        active_pair = get_active_pair()
    except Exception:
        active_pair = None
    ticker = pair_to_ticker(active_pair) or "PAXG"
    param_file_name = f"avellaneda_parameters_{ticker}.json"
    logger.info(f"Resolving parameters for pair '{active_pair or 'unknown'}' (ticker '{ticker}') using {params_dir}")

    params_file = params_dir / param_file_name

    if params_file.exists():
        try:
            params_MM = json.loads(params_file.read_text(encoding='utf-8'))
            logger.info(f"Successfully loaded parameters from: {params_file}")
            return params_MM
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading {params_file}: {e}")

    if start_dir is None:
        try:
            start_dir = Path(__file__).resolve().parent
        except NameError:  # e.g., interactive
            start_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()

    search_locations = [
        f"scripts/{param_file_name}",
        param_file_name,
        f"user_data/strategies/{param_file_name}",
        f"../scripts/{param_file_name}",
        f"../../scripts/{param_file_name}"
    ]

    for location in search_locations:
        try:
            params_file_found = find_upwards(location, start_dir, max_up)
            params_MM = json.loads(params_file_found.read_text(encoding='utf-8'))
            logger.info(f"Successfully loaded parameters from: {params_file_found}")
            return params_MM
        except FileNotFoundError:
            continue
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading {location}: {e}")
            continue

    raise FileNotFoundError(
        f"Could not find '{param_file_name}' in any of these locations:\n"
        f"  - Primary: {params_file}\n"
        f"  - " + "\n  - ".join(str(start_dir / loc) for loc in search_locations)
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

    # Minimum number of data collection periods required before trading
    min_data_periods: int = 3
    
    minimal_roi = {
        "0": -1
    }

    params_MM = None
    gamma = None
    k_bid = None
    k_ask = None
    sigma = None
    time_horizon_hours = None

    fees_HL_maker = 0.02/100.0

    nb_loop = 0

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
        if len(pairs)!=1:
            sys.exit()

        if not self.can_short:
            logger.info('Running calculation of parameters')
            symbol = pairs[0].replace("/USDC:USDC","")
            logger.info(f"Current symbol: {symbol}")
            run_avellaneda_param_calculation()

        self.params_MM = load_configs()
        self.gamma = self.params_MM['optimal_parameters']['gamma']
        self.k_bid = self.params_MM['market_data'].get('k_bid', self.params_MM['market_data'].get('k'))
        self.k_ask = self.params_MM['market_data'].get('k_ask', self.params_MM['market_data'].get('k'))
        self.sigma = self.params_MM['market_data']['sigma']
        # Default to 0.5 hours if not present in older JSONs
        self.time_horizon_hours = self.params_MM['optimal_parameters'].get('time_horizon_hours', 0.5)

        # Check if we have sufficient data periods
        num_periods = self.params_MM.get('current_state', {}).get('num_data_periods', 0)
        if num_periods < self.min_data_periods:
            logger.warning(f"Insufficient data collected: {num_periods}/{self.min_data_periods} periods. Trading will be disabled until more data is collected.")
        else:
            logger.info(f"Sufficient data available: {num_periods} periods collected.")
        

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop). For each loop, it will run populate_indicators on all pairs.
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """

        if not self.can_short:
            if self.nb_loop%10==0:
                logger.info('Running calculation of parameters')
                run_avellaneda_param_calculation()
            self.nb_loop = self.nb_loop + 1

        self.params_MM = load_configs()
        self.gamma = self.params_MM['optimal_parameters']['gamma']
        self.k_bid = self.params_MM['market_data'].get('k_bid', self.params_MM['market_data'].get('k'))
        self.k_ask = self.params_MM['market_data'].get('k_ask', self.params_MM['market_data'].get('k'))
        self.sigma = self.params_MM['market_data']['sigma']
        self.time_horizon_hours = self.params_MM['optimal_parameters'].get('time_horizon_hours', 0.5)

    def informative_pairs(self):
        """
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Only allow entry if we have sufficient data collected by the data collector.
        """
        # Check if parameters are loaded and we have enough collected data periods
        if self.params_MM is not None and self.sigma is not None:
            # Check if we have enough data periods from the data collector
            num_periods = self.params_MM.get('current_state', {}).get('num_data_periods', 0)
            if num_periods >= self.min_data_periods:
                dataframe.loc[:, 'enter_long'] = 1
            else:
                logger.warning(f"Insufficient data periods: {num_periods}/{self.min_data_periods}. Waiting for more data collection.")
                dataframe.loc[:, 'enter_long'] = 0
        else:
            dataframe.loc[:, 'enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
    
    def get_mid_price(self, pair: str, fallback_rate: float) -> float:
        """
        Get mid price from orderbook, fallback to provided rate if orderbook unavailable
        """
        orderbook = self.dp.orderbook(pair, maximum=1)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            return (best_bid + best_ask) / 2
        else:
            return fallback_rate
     
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:

        if self.sigma is None or self.gamma is None or self.params_MM is None:
            return None

        if side!="long":
            return None
        
        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC","")
        open_trades = Trade.get_open_trades()
        total_quote_position = sum([float(trade.open_rate) * float(trade.amount) for trade in open_trades])
        total_capital = self.wallets.get_total(self.config['stake_currency'])
        q_inventory_exposure = total_quote_position / total_capital if total_capital > 0 else 0
        r_buy, r_sell = calculate_optimal_spreads(mid_price,self.sigma,self.k_bid,self.k_ask,self.gamma,self.time_horizon_hours/24.0,q_inventory_exposure,self.fees_HL_maker)

        return r_buy

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:
        
        if self.sigma is None or self.gamma is None or self.params_MM is None:
            return None

        if trade.is_short:
            return None

        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC","")
        open_trades = Trade.get_open_trades()
        total_quote_position = sum([float(trade.open_rate) * float(trade.amount) for trade in open_trades])
        total_capital = self.wallets.get_total(self.config['stake_currency'])
        q_inventory_exposure = total_quote_position / total_capital if total_capital > 0 else 0
        r_buy, r_sell = calculate_optimal_spreads(mid_price,self.sigma,self.k_bid,self.k_ask,self.gamma,self.time_horizon_hours/24.0,q_inventory_exposure,self.fees_HL_maker)

        return r_sell

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: str, side: str, **kwargs) -> float:
        
        if self.sigma is None or self.gamma is None or self.params_MM is None:
            return None

        if trade.is_short:
            return None
        
        mid_price = self.get_mid_price(pair, proposed_rate)
        symbol = pair.replace("/USDC:USDC","")
        open_trades = Trade.get_open_trades()
        total_quote_position = sum([float(trade.open_rate) * float(trade.amount) for trade in open_trades])
        total_capital = self.wallets.get_total(self.config['stake_currency'])
        q_inventory_exposure = total_quote_position / total_capital if total_capital > 0 else 0
        r_buy, r_sell = calculate_optimal_spreads(mid_price,self.sigma,self.k_bid,self.k_ask,self.gamma,self.time_horizon_hours/24.0,q_inventory_exposure,self.fees_HL_maker)

        return r_buy

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
