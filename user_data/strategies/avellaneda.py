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

def calculate_optimal_spreads(mid_price,sigma,k,gamma,time_remaining,q_inventory_exposure, fee):
    # Calculate base spread using Avellaneda-Stoikov formula
    spread_base = gamma * sigma**2.0 * time_remaining + (2.0 / gamma) * np.log(1.0 + (gamma / k))
    half_spread = spread_base / 2.0

    # Calculate inventory penalty
    print(q_inventory_exposure * gamma * sigma**2.0 * time_remaining)

    # Calculate reservation price (our "fair value" given inventory and time remaining)
    r = mid_price - q_inventory_exposure * gamma * sigma**2.0 * time_remaining

    # Calculate gap between reservation price and current mid-price
    gap = abs(r - mid_price)

    # Apply inventory adjustment to spreads
    # When long (positive inventory): tighten ask to sell faster, widen bid to buy slower
    # When short (negative inventory): widen ask to sell slower, tighten bid to buy faster
    if r >= mid_price:  # Reservation price above mid (short inventory, want to buy)
        delta_a = half_spread + gap  # Widen ask (sell at higher price)
        delta_b = half_spread - gap  # Tighten bid (buy more aggressively)
    else:       # Reservation price below mid (long inventory, want to sell)
        delta_a = half_spread - gap  # Tighten ask (sell more aggressively)
        delta_b = half_spread + gap  # Widen bid (buy at lower price)

    # Calculate limit order prices
    r_b = r - delta_b - mid_price*fee
    r_a = r + delta_a + mid_price*fee

    # Calculate relative percentages
    delta_a_percent = (abs(r_b-mid_price) / mid_price) * 100.0
    delta_b_percent = (abs(r_b-mid_price) / mid_price) * 100.0

    mm_logger.info("=" * 65)
    mm_logger.info("📊 AVELLANEDA-STOIKOV MODEL PARAMETERS")
    mm_logger.info("=" * 65)
    mm_logger.info(f"│ Time Remaining Fraction    │ {time_remaining:>12.4f}          │")
    mm_logger.info(f"│ Inventory Exposure         │ {q_inventory_exposure:>12.4f}          │")
    mm_logger.info(f"│ Sigma (Volatility)         │ {sigma:>12.4f}          │")
    mm_logger.info(f"│ K                          │ {k:>12.4f}          │")
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

def fraction_of_day_remaining_utc():
    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    
    # Get end of day (midnight) in UTC
    end_of_day = now_utc.replace(hour=23, minute=59, second=59, microsecond=999999)
    # Or more precisely, start of next day:
    # end_of_day = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate total seconds in a day
    total_seconds_in_day = 24 * 60 * 60
    
    # Calculate seconds remaining
    seconds_remaining = (end_of_day - now_utc).total_seconds()
    
    # Calculate fraction remaining
    fraction_remaining = seconds_remaining / total_seconds_in_day
    
    return fraction_remaining

#---------------------------------------------------------- LOAD CONFIG ----------------------------------------------------------

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
    if start_dir is None:
        try:
            start_dir = Path(__file__).resolve().parent
        except NameError:  # e.g., interactive
            start_dir = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else Path.cwd()

    # List of potential file locations to check (in order of preference)
    search_locations = [
        "scripts/avellaneda_parameters_BTC.json",  # Most likely location
        "avellaneda_parameters_BTC.json",          # Root directory
        "user_data/strategies/avellaneda_parameters_BTC.json",  # Same directory
        "../scripts/avellaneda_parameters_BTC.json",  # One level up then scripts
        "../../scripts/avellaneda_parameters_BTC.json"  # Two levels up then scripts
    ]
    
    # First try to find the file using the existing upward search for each location
    for location in search_locations:
        try:
            params_file = find_upwards(location, start_dir, max_up)
            params_MM = json.loads(params_file.read_text(encoding="utf-8"))
            print(f"Successfully loaded parameters from: {params_file}")
            return params_MM
        except FileNotFoundError:
            continue
    
    # If upward search fails, try direct relative paths from start directory
    for location in search_locations:
        try:
            # Try relative to start directory
            params_path = start_dir / location
            if params_path.exists():
                params_MM = json.loads(params_path.read_text(encoding="utf-8"))
                print(f"Successfully loaded parameters from: {params_path}")
                return params_MM
                
            # Try relative to project root (go up to find project root)
            current = start_dir
            for _ in range(max_up):
                potential_root = current / location
                if potential_root.exists():
                    params_MM = json.loads(potential_root.read_text(encoding="utf-8"))
                    print(f"Successfully loaded parameters from: {potential_root}")
                    return params_MM
                if current.parent == current:
                    break
                current = current.parent
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {location}: {e}")
            continue
    
    # If all attempts fail, provide helpful error message
    raise FileNotFoundError(
        f"Could not find 'avellaneda_parameters_BTC.json' in any of these locations:\n"
        f"  - {chr(10).join([str(start_dir / loc) for loc in search_locations])}\n"
        f"Please ensure the file exists and is accessible."
    )

#---------------------------------------------------------- LOAD CONFIG ----------------------------------------------------------

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

    minimal_roi = {
        "0": -1
    }

    params_MM = None
    gamma = None
    k = None
    sigma = None

    fees_HL_maker = 0.015/100.0

    nb_loop = 0

    stoploss = -0.75

    trailing_stop = False

    timeframe = '5m'

    startup_candle_count: int = 0

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
        self.k = self.params_MM['market_data']['k']
        self.sigma = self.params_MM['market_data']['sigma']
        

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
        self.k = self.params_MM['market_data']['k']
        self.sigma = self.params_MM['market_data']['sigma']

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
        """
        if self.params_MM is not None and self.sigma is not None:
            dataframe.loc[:, 'enter_long'] = 1
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
        q_inventory_exposure = total_quote_position / float(self.config['stake_amount'])
        r_buy, r_sell = calculate_optimal_spreads(mid_price,self.sigma,self.k,self.gamma,fraction_of_day_remaining_utc(),q_inventory_exposure,self.fees_HL_maker)

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
        q_inventory_exposure = total_quote_position / float(self.config['stake_amount'])
        r_buy, r_sell = calculate_optimal_spreads(mid_price,self.sigma,self.k,self.gamma,fraction_of_day_remaining_utc(),q_inventory_exposure,self.fees_HL_maker)

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
        q_inventory_exposure = total_quote_position / float(self.config['stake_amount'])
        r_buy, r_sell = calculate_optimal_spreads(mid_price,self.sigma,self.k,self.gamma,fraction_of_day_remaining_utc(),q_inventory_exposure,self.fees_HL_maker)

        return r_buy

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        return "always_exit"

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
