#!/usr/bin/env python3
"""
Hyperliquid Tick Data Collector (Fixed Version)
Collects time-tagged prices, executed orders, and order book data via websockets

Fixes applied:
1. Stats counter bug - accepts count parameter
2. Symbol KeyError risk - validates symbols before access
3. Missing symbol field in output data - added to all records
4. Thread safety for writer access - added writer_lock
5. Race condition during reconnection - added reconnecting state
6. Silent data drop on buffer overflow - logs warnings at 80% capacity
7. File handle leak on writer creation failure - proper cleanup
8. tell() unreliable for compressed file size - uses row count instead
9. Inconsistent logging - all output uses logging module
10. Daemon threads issue - non-daemon threads with proper join
11. Periodic tasks sleep timing - checks running flag first
12. Removed unused TickData base class
"""

import json
import time
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import os
import threading
import random
import logging
import signal
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hyperliquid.info import Info
from hyperliquid.utils import constants


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PriceData:
    """Price tick data"""
    timestamp: float
    symbol: str
    price: float
    size: float
    exchange_timestamp: Optional[int] = None
    side: Optional[str] = None  # 'bid' or 'ask' for BBO data


@dataclass
class TradeData:
    """Trade execution data"""
    timestamp: float
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    exchange_timestamp: Optional[int] = None
    trade_id: Optional[str] = None


@dataclass
class OrderBookLevel:
    """Order book level data"""
    price: float
    size: float


@dataclass
class OrderBookData:
    """Order book snapshot data"""
    timestamp: float
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    exchange_timestamp: Optional[int] = None
    sequence: Optional[int] = None


# =============================================================================
# Statistics Tracker
# =============================================================================

class DataStats:
    """Track data collection statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.counters = defaultdict(int)
        self.last_update = time.time()
        self.recent_data = defaultdict(lambda: deque(maxlen=100))
        self.reconnection_count = 0
        self.last_data_time = time.time()
        self._lock = threading.Lock()  # FIX: Thread-safe counter updates
    
    def update(self, data_type: str, count: int = 1, data: Any = None):
        """
        Update statistics counters.
        
        FIX #1: Now accepts a count parameter to properly track batch updates.
        
        Args:
            data_type: Type of data being tracked
            count: Number of items to add to counter (default: 1)
            data: Optional data sample for recent_data tracking
        """
        with self._lock:
            self.counters[data_type] += count  # FIX: Use count parameter
            self.last_update = time.time()
            self.last_data_time = time.time()
            if data:
                self.recent_data[data_type].append(data)
    
    def record_reconnection(self):
        with self._lock:
            self.reconnection_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            runtime = time.time() - self.start_time
            return {
                'runtime_seconds': runtime,
                'runtime_formatted': f"{runtime//3600:.0f}h {(runtime%3600)//60:.0f}m {runtime%60:.0f}s",
                'counters': dict(self.counters),
                'rates_per_minute': {k: v / (runtime / 60) for k, v in self.counters.items() if runtime > 0},
                'last_update': datetime.fromtimestamp(self.last_update).strftime('%H:%M:%S'),
                'reconnections': self.reconnection_count,
                'seconds_since_last_data': time.time() - self.last_data_time
            }


# =============================================================================
# Main Collector Class
# =============================================================================

class HyperliquidDataCollector:
    """Main data collector class"""
    
    # Buffer capacity warning threshold (80%)
    BUFFER_WARNING_THRESHOLD = 0.8
    
    def __init__(self, symbols: List[str], output_dir: str = "data", orderbook_depth: int = 20, rotation_interval: int = 900):
        self.symbols = [s.upper() for s in symbols]  # FIX: Normalize to uppercase
        self.output_dir = output_dir
        self.orderbook_depth = orderbook_depth
        self.rotation_interval = rotation_interval  # Default 15 minutes

        self.info = None
        self.stats = DataStats()
        self.subscription_ids = []
        self.connection_healthy = False
        self.is_reconnecting = False  # FIX #5: Track reconnection state
        self.reconnection_delay = 1
        self.max_reconnection_delay = 300
        self.data_timeout = 60
        
        # Buffer configuration
        self.price_buffer_size = 10000
        self.trade_buffer_size = 10000
        self.orderbook_buffer_size = 1000
        
        # Create separate buffers for each symbol
        self.data_buffers = {}
        for symbol in self.symbols:
            self.data_buffers[symbol] = {
                'prices': deque(maxlen=self.price_buffer_size),
                'trades': deque(maxlen=self.trade_buffer_size),
                'orderbooks': deque(maxlen=self.orderbook_buffer_size)
            }
        
        self.running = False
        
        # FIX #10: Track threads for proper shutdown
        self.flush_thread = None
        self.summary_thread = None
        self.reconnection_thread = None
        
        # Concurrency control
        self.data_lock = threading.Lock()  # For data buffers
        self.writer_lock = threading.Lock()  # FIX #4: Separate lock for writers
        
        # Parquet writing state - uses row count instead of file size
        self.active_writers = {}  # (symbol, data_type) -> {'writer': ..., 'row_count': int, ...}
        self.target_row_count = 100000  # FIX #8: Use row count instead of compressed file size
        self.target_file_size = 10 * 1024 * 1024  # Keep for reference/logging
        
        # Setup logging - FIX #9: Configure properly
        self._setup_logging()
        
        # Directory paths for each symbol
        self.symbol_dirs = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Data Storage
        self._init_data_storage()
        
        # Initialize connection
        self._init_connection()
    
    def _setup_logging(self):
        """Configure logging with consistent format"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_data_storage(self):
        """Initialize directories for data storage - separate folders per symbol"""
        for symbol in self.symbols:
            self.symbol_dirs[symbol] = {}
            
            # Price data directory
            price_dir = os.path.join(self.output_dir, f"prices_{symbol}.parquet")
            self.symbol_dirs[symbol]['prices'] = price_dir
            os.makedirs(price_dir, exist_ok=True)
            
            # Trade data directory
            trade_dir = os.path.join(self.output_dir, f"trades_{symbol}.parquet")
            self.symbol_dirs[symbol]['trades'] = trade_dir
            os.makedirs(trade_dir, exist_ok=True)
            
            # Order book directory
            orderbook_dir = os.path.join(self.output_dir, f"orderbooks_{symbol}.parquet")
            self.symbol_dirs[symbol]['orderbooks'] = orderbook_dir
            os.makedirs(orderbook_dir, exist_ok=True)
    
    def _init_connection(self):
        """Initialize websocket connection"""
        try:
            self.info = Info(constants.MAINNET_API_URL, skip_ws=False)
            self.connection_healthy = True
            self.logger.info("Websocket connection initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection: {e}")
            self.connection_healthy = False
    
    def _is_connection_healthy(self) -> bool:
        """Check if connection is healthy based on recent data"""
        if not self.connection_healthy:
            return False
        
        time_since_last_data = time.time() - self.stats.last_data_time
        if time_since_last_data > self.data_timeout:
            self.logger.warning(f"No data received for {time_since_last_data:.1f} seconds - connection may be dead")
            return False
        
        return True
    
    def _validate_symbol(self, symbol: str) -> Optional[str]:
        """
        FIX #2: Validate and normalize symbol before use.
        
        Returns normalized symbol if valid, None if invalid.
        """
        if symbol is None:
            return None
        
        normalized = symbol.upper()
        
        if normalized not in self.data_buffers:
            self.logger.warning(f"Received data for unconfigured symbol: {symbol}")
            return None
        
        return normalized
    
    def _check_buffer_capacity(self, symbol: str, buffer_type: str):
        """
        FIX #6: Log warning when buffer approaches capacity.
        """
        buffer = self.data_buffers[symbol][buffer_type]
        capacity = buffer.maxlen
        current_size = len(buffer)
        
        if current_size >= capacity * self.BUFFER_WARNING_THRESHOLD:
            self.logger.warning(
                f"Buffer near capacity for {symbol}/{buffer_type}: "
                f"{current_size}/{capacity} ({current_size/capacity*100:.1f}%)"
            )
    
    def _reconnect(self):
        """Reconnect to websocket with exponential backoff"""
        self.logger.info(f"Attempting reconnection (delay: {self.reconnection_delay}s)")
        self.is_reconnecting = True  # FIX #5: Set reconnecting state
        
        # Wait before reconnecting
        time.sleep(self.reconnection_delay)
        
        try:
            # Cleanup old connection
            if self.info:
                try:
                    self.info.disconnect_websocket()
                except Exception:
                    pass
            
            # Create new connection
            self._init_connection()
            
            if self.connection_healthy:
                # FIX #5: Clear subscriptions atomically before resubscribing
                with self.data_lock:
                    self.subscription_ids.clear()
                
                self._subscribe_to_feeds()
                
                # Reset reconnection delay on successful connection
                self.reconnection_delay = 1
                self.stats.record_reconnection()
                self.is_reconnecting = False  # FIX #5: Clear reconnecting state
                self.logger.info("Successfully reconnected and resubscribed to feeds")
                return True
            
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
        
        self.is_reconnecting = False
        # Increase delay for next attempt (exponential backoff)
        self.reconnection_delay = min(
            self.reconnection_delay * 2 + random.uniform(0, 1),
            self.max_reconnection_delay
        )
        return False
    
    def _subscribe_to_feeds(self):
        """Subscribe to all data feeds"""
        for symbol in self.symbols:
            self.logger.info(f"Subscribing to data feeds for {symbol}...")
            
            # Subscribe to best bid/offer
            bbo_id = self.info.subscribe(
                {"type": "bbo", "coin": symbol},
                self._handle_bbo_data
            )
            self.subscription_ids.append(bbo_id)
            
            # Subscribe to trades
            trades_id = self.info.subscribe(
                {"type": "trades", "coin": symbol},
                self._handle_trade_data
            )
            self.subscription_ids.append(trades_id)
            
            # Subscribe to order book
            l2book_id = self.info.subscribe(
                {"type": "l2Book", "coin": symbol},
                self._handle_orderbook_data
            )
            self.subscription_ids.append(l2book_id)
        
        self.logger.info(f"Subscribed to {len(self.subscription_ids)} data feeds")
    
    def _connection_monitor(self):
        """Monitor connection health and reconnect if needed"""
        while self.running:
            # FIX #11: Check running flag before and after sleep
            for _ in range(10):  # Check every second, total 10 seconds
                if not self.running:
                    return
                time.sleep(1)
            
            if not self.running:
                break
            
            if not self._is_connection_healthy() and not self.is_reconnecting:
                self.connection_healthy = False
                self.logger.warning("Connection unhealthy - attempting reconnection")
                
                # Try to reconnect
                while self.running and not self._reconnect():
                    if not self.running:
                        break
    
    def _get_or_create_writer(self, symbol: str, data_type: str, dir_path: str, schema: pa.Schema):
        """
        Get an existing ParquetWriter or create a new one if needed.
        
        FIX #7: Proper file handle cleanup on failure
        FIX #8: Uses row count instead of file size for rotation
        """
        key = (symbol, data_type)
        
        # Check if we have an active writer
        if key in self.active_writers:
            writer_info = self.active_writers[key]
            
            # Check for rotation (row count OR time)
            time_since_creation = time.time() - writer_info.get('creation_time', 0)
            
            # FIX #8: Check row count instead of compressed file size
            if (writer_info['row_count'] >= self.target_row_count or 
                time_since_creation >= self.rotation_interval):
                
                reason = "rows" if writer_info['row_count'] >= self.target_row_count else "time"
                self.logger.info(f"Rotating file for {symbol}/{data_type} (reason: {reason}, rows: {writer_info['row_count']}, age: {time_since_creation:.1f}s)")
                self._close_writer(symbol, data_type)
            else:
                return writer_info['writer']
        
        # Create new writer
        filename = f"part_{int(time.time() * 1000)}.parquet"
        file_path = os.path.join(dir_path, filename)
        
        file_handle = None
        try:
            # FIX #7: Track file_handle for cleanup on failure
            file_handle = open(file_path, 'wb')
            writer = pq.ParquetWriter(file_handle, schema, compression='zstd')
            
            self.active_writers[key] = {
                'writer': writer,
                'file_handle': file_handle,
                'path': file_path,
                'row_count': 0,  # FIX #8: Track row count
                'creation_time': time.time()  # Track creation time for rotation
            }
            self.logger.info(f"Created new parquet file: {file_path}")
            return writer
            
        except Exception as e:
            # FIX #7: Clean up file handle on failure
            self.logger.error(f"Failed to create parquet writer for {file_path}: {e}")
            if file_handle is not None:
                try:
                    file_handle.close()
                except Exception:
                    pass
                # Remove partial file
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            return None

    def _close_writer(self, symbol: str, data_type: str):
        """Close an active writer"""
        key = (symbol, data_type)
        if key not in self.active_writers:
            return
        
        writer_info = self.active_writers[key]
        try:
            # Flush before closing
            try:
                f = writer_info['file_handle']
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                self.logger.error(f"Error flushing before close for {key}: {e}")

            # Close writer first (writes footer)
            writer_info['writer'].close()
            # Close file handle
            writer_info['file_handle'].close()
            
            # Log final stats
            self.logger.info(
                f"Closed parquet file: {writer_info['path']} "
                f"(rows: {writer_info['row_count']})"
            )
        except Exception as e:
            self.logger.error(f"Error closing writer for {key}: {e}")
            # Try to at least close the file handle
            try:
                writer_info['file_handle'].close()
            except Exception:
                pass
        finally:
            # Always delete the entry
            del self.active_writers[key]

    def _write_buffered_data(self, symbol: str, data_type: str, data: List[Dict], dir_path: str):
        """Write buffered data to Parquet file using a persistent writer"""
        if not data:
            return

        try:
            # Convert to DataFrame
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([asdict(item) for item in data])
            
            # Convert to Arrow Table
            table = pa.Table.from_pandas(df)
            row_count = len(df)
            
            # FIX #4: Use writer lock for thread safety
            with self.writer_lock:
                # Get or create writer
                writer = self._get_or_create_writer(symbol, data_type, dir_path, table.schema)
                
                if writer:
                    writer.write_table(table)
                    
                    # Update row count
                    key = (symbol, data_type)
                    if key in self.active_writers:
                        self.active_writers[key]['row_count'] += row_count
                        
                        # Force flush to disk
                        writer_info = self.active_writers[key]
                        f = writer_info['file_handle']
                        f.flush()
                        os.fsync(f.fileno())

        except Exception as e:
            self.logger.error(f"Error writing to Parquet {dir_path}: {e}")
    
    def _handle_bbo_data(self, data: Dict[str, Any]):
        """Handle best bid/offer data"""
        # FIX #5: Ignore data during reconnection
        if self.is_reconnecting:
            return
        
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'bbo' and 'data' in data:
                bbo_data = data['data']
                raw_symbol = bbo_data.get('coin')
                
                # FIX #2: Validate symbol
                symbol = self._validate_symbol(raw_symbol)
                if symbol is None:
                    return
                
                # BBO format: bbo array with [bid, ask]
                if 'bbo' in bbo_data and len(bbo_data['bbo']) >= 2:
                    bid_info = bbo_data['bbo'][0]
                    ask_info = bbo_data['bbo'][1]
                    
                    # FIX #3: Include symbol in output data
                    bid_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': float(bid_info['px']),
                        'size': float(bid_info['sz']),
                        'side': 'bid',
                        'exchange_timestamp': bbo_data.get('time')
                    }
                    
                    ask_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': float(ask_info['px']),
                        'size': float(ask_info['sz']),
                        'side': 'ask',
                        'exchange_timestamp': bbo_data.get('time')
                    }
                    
                    with self.data_lock:
                        self._check_buffer_capacity(symbol, 'prices')  # FIX #6
                        self.data_buffers[symbol]['prices'].append(bid_data)
                        self.data_buffers[symbol]['prices'].append(ask_data)
                
                self.stats.update('bbo_updates')
                
            else:
                # Direct format fallback
                raw_symbol = data.get('coin')
                symbol = self._validate_symbol(raw_symbol)
                if symbol is None:
                    return
                
                if 'bid' in data and data['bid']:
                    bid_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': float(data['bid']['px']),
                        'size': float(data['bid']['sz']),
                        'side': 'bid',
                        'exchange_timestamp': data.get('time')
                    }
                    with self.data_lock:
                        self._check_buffer_capacity(symbol, 'prices')
                        self.data_buffers[symbol]['prices'].append(bid_data)
                
                if 'ask' in data and data['ask']:
                    ask_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': float(data['ask']['px']),
                        'size': float(data['ask']['sz']),
                        'side': 'ask',
                        'exchange_timestamp': data.get('time')
                    }
                    with self.data_lock:
                        self._check_buffer_capacity(symbol, 'prices')
                        self.data_buffers[symbol]['prices'].append(ask_data)
                
                self.stats.update('bbo_updates')
                
        except Exception as e:
            self.logger.error(f"Error handling BBO data: {e}")  # FIX #9
    
    def _handle_trade_data(self, data: Dict[str, Any]):
        """Handle trade data"""
        if self.is_reconnecting:
            return
        
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'trades' and 'data' in data:
                trades = data['data']
                trades_processed = 0
                
                for trade in trades:
                    raw_symbol = trade.get('coin')
                    symbol = self._validate_symbol(raw_symbol)
                    if symbol is None:
                        continue
                    
                    # Convert A/B to buy/sell
                    side = 'sell' if trade['side'] == 'A' else 'buy'
                    
                    # FIX #3: Include symbol in output
                    trade_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': side,
                        'trade_id': str(trade.get('tid')),
                        'exchange_timestamp': trade.get('time')
                    }
                    
                    with self.data_lock:
                        self._check_buffer_capacity(symbol, 'trades')
                        self.data_buffers[symbol]['trades'].append(trade_data)
                    
                    trades_processed += 1
                
                # FIX #1: Pass actual count to stats
                self.stats.update('trades', count=trades_processed)
                
            else:
                # Direct format fallback
                if isinstance(data, list):
                    trades = data
                else:
                    trades = [data]
                
                trades_processed = 0
                for trade in trades:
                    raw_symbol = trade.get('coin')
                    symbol = self._validate_symbol(raw_symbol)
                    if symbol is None:
                        continue
                    
                    side = 'sell' if trade['side'] == 'A' else 'buy'
                    
                    trade_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': side,
                        'trade_id': str(trade.get('tid')),
                        'exchange_timestamp': trade.get('time')
                    }
                    
                    with self.data_lock:
                        self._check_buffer_capacity(symbol, 'trades')
                        self.data_buffers[symbol]['trades'].append(trade_data)
                    
                    trades_processed += 1
                
                self.stats.update('trades', count=trades_processed)
                
        except Exception as e:
            self.logger.error(f"Error handling trade data: {e}")
    
    def _handle_orderbook_data(self, data: Dict[str, Any]):
        """Handle order book data"""
        if self.is_reconnecting:
            return
        
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'l2Book' and 'data' in data:
                book_data = data['data']
                raw_symbol = book_data.get('coin')
                symbol = self._validate_symbol(raw_symbol)
                if symbol is None:
                    return
                
                # Parse bids and asks
                bids = []
                asks = []
                
                if 'levels' in book_data and len(book_data['levels']) >= 2:
                    bids_array = book_data['levels'][0]
                    asks_array = book_data['levels'][1]
                    
                    for bid in bids_array:
                        bids.append(OrderBookLevel(price=float(bid['px']), size=float(bid['sz'])))
                    
                    for ask in asks_array:
                        asks.append(OrderBookLevel(price=float(ask['px']), size=float(ask['sz'])))
                
                # FIX #3: Include symbol in flattened output
                csv_row = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'sequence': book_data.get('time'),
                    'exchange_timestamp': book_data.get('time')
                }
                
                for i in range(self.orderbook_depth):
                    if i < len(bids):
                        csv_row[f'bid_price_{i}'] = bids[i].price
                        csv_row[f'bid_size_{i}'] = bids[i].size
                    else:
                        csv_row[f'bid_price_{i}'] = None
                        csv_row[f'bid_size_{i}'] = None
                    
                    if i < len(asks):
                        csv_row[f'ask_price_{i}'] = asks[i].price
                        csv_row[f'ask_size_{i}'] = asks[i].size
                    else:
                        csv_row[f'ask_price_{i}'] = None
                        csv_row[f'ask_size_{i}'] = None
                
                with self.data_lock:
                    self._check_buffer_capacity(symbol, 'orderbooks')
                    self.data_buffers[symbol]['orderbooks'].append(csv_row)
                
                self.stats.update('orderbook_updates')
                
            else:
                # Direct format fallback
                raw_symbol = data.get('coin')
                symbol = self._validate_symbol(raw_symbol)
                if symbol is None:
                    return
                
                bids = []
                asks = []
                
                if 'levels' in data and len(data['levels']) >= 2:
                    bids_array = data['levels'][0]
                    asks_array = data['levels'][1]
                    
                    for bid in bids_array:
                        bids.append(OrderBookLevel(price=float(bid['px']), size=float(bid['sz'])))
                    
                    for ask in asks_array:
                        asks.append(OrderBookLevel(price=float(ask['px']), size=float(ask['sz'])))
                
                csv_row = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'sequence': data.get('time'),
                    'exchange_timestamp': data.get('time')
                }
                
                for i in range(self.orderbook_depth):
                    if i < len(bids):
                        csv_row[f'bid_price_{i}'] = bids[i].price
                        csv_row[f'bid_size_{i}'] = bids[i].size
                    else:
                        csv_row[f'bid_price_{i}'] = None
                        csv_row[f'bid_size_{i}'] = None
                    
                    if i < len(asks):
                        csv_row[f'ask_price_{i}'] = asks[i].price
                        csv_row[f'ask_size_{i}'] = asks[i].size
                    else:
                        csv_row[f'ask_price_{i}'] = None
                        csv_row[f'ask_size_{i}'] = None
                
                with self.data_lock:
                    self._check_buffer_capacity(symbol, 'orderbooks')
                    self.data_buffers[symbol]['orderbooks'].append(csv_row)
                
                self.stats.update('orderbook_updates')
            
        except Exception as e:
            self.logger.error(f"Error handling order book data: {e}")
    
    def _flush_buffers(self):
        """Flush data buffers to Parquet files - separate directories per symbol"""
        try:
            for symbol in self.symbols:
                symbol_buffers = self.data_buffers[symbol]
                symbol_dirs = self.symbol_dirs[symbol]
                
                # Use lock to safely move data out of buffers
                with self.data_lock:
                    prices_to_write = list(symbol_buffers['prices'])
                    symbol_buffers['prices'].clear()
                    
                    trades_to_write = list(symbol_buffers['trades'])
                    symbol_buffers['trades'].clear()
                    
                    orderbooks_to_write = list(symbol_buffers['orderbooks'])
                    symbol_buffers['orderbooks'].clear()
                
                # Write outside the lock
                if prices_to_write:
                    self._write_buffered_data(symbol, 'prices', prices_to_write, symbol_dirs['prices'])
                
                if trades_to_write:
                    self._write_buffered_data(symbol, 'trades', trades_to_write, symbol_dirs['trades'])
                
                if orderbooks_to_write:
                    self._write_buffered_data(symbol, 'orderbooks', orderbooks_to_write, symbol_dirs['orderbooks'])
                
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")
    
    def _print_summary(self):
        """Print data collection summary"""
        summary = self.stats.get_summary()
        
        lines = [
            "",
            "=" * 60,
            f"DATA COLLECTION SUMMARY - {summary['last_update']}",
            "=" * 60,
            f"Runtime: {summary['runtime_formatted']}",
            f"Connection: {'🟢 Healthy' if self.connection_healthy else '🔴 Reconnecting'}",
        ]
        
        if summary['reconnections'] > 0:
            lines.append(f"Reconnections: {summary['reconnections']}")
        
        lines.append(f"Time since last data: {summary['seconds_since_last_data']:.1f}s")
        lines.append("Data collected:")
        
        for data_type, count in summary['counters'].items():
            rate = summary['rates_per_minute'].get(data_type, 0)
            lines.append(f"  {data_type}: {count:,} ({rate:.1f}/min)")
        
        lines.append("")
        lines.append("Buffer sizes by symbol:")
        
        for symbol in self.symbols:
            symbol_buffers = self.data_buffers[symbol]
            total_buffered = sum(len(buffer) for buffer in symbol_buffers.values())
            lines.append(
                f"  {symbol}: {total_buffered} "
                f"({len(symbol_buffers['prices'])} prices, "
                f"{len(symbol_buffers['trades'])} trades, "
                f"{len(symbol_buffers['orderbooks'])} orderbooks)"
            )
        
        # Show active writer stats
        with self.writer_lock:
            lines.append(f"Active Parquet Writers: {len(self.active_writers)}")
            for key, info in self.active_writers.items():
                lines.append(f"  {key[0]}/{key[1]}: {info['row_count']} rows")
        
        lines.append("=" * 60)
        
        self.logger.info("\n".join(lines))
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals from Docker (SIGTERM) and Ctrl+C (SIGINT)"""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.logger.info(f"{signal_name} received - initiating graceful shutdown...")
        self.stop_collection()
        sys.exit(0)

    def start_collection(self):
        """Start data collection"""
        self.logger.info(f"Starting Hyperliquid data collection for symbols: {self.symbols}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Target rows per file: {self.target_row_count:,}")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.logger.info("Signal handlers registered (SIGTERM, SIGINT)")

        self.running = True

        try:
            # Subscribe to data feeds
            if self.connection_healthy:
                self._subscribe_to_feeds()
            else:
                self.logger.error("Initial connection failed - will attempt reconnection")
            
            # FIX #10: Use non-daemon threads for proper shutdown
            self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=False)
            self.flush_thread.start()
            
            self.summary_thread = threading.Thread(target=self._periodic_summary, daemon=False)
            self.summary_thread.start()
            
            self.reconnection_thread = threading.Thread(target=self._connection_monitor, daemon=False)
            self.reconnection_thread.start()
            
            self.logger.info("Data collection started. Press Ctrl+C or 'docker compose down' to stop.")
            self.logger.info("Automatic reconnection enabled - connection drops will be handled automatically.")
            
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received...")
            self.stop_collection()
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            self.stop_collection()
    
    def _periodic_flush(self):
        """Periodically flush buffers to disk"""
        while self.running:
            # FIX #11: Check running flag with shorter sleep intervals
            for _ in range(5):  # 5 x 1 second = 5 seconds total
                if not self.running:
                    return
                time.sleep(1)
            
            if self.running:
                self._flush_buffers()
    
    def _periodic_summary(self):
        """Periodically print collection summary"""
        while self.running:
            # FIX #11: Check running flag with shorter sleep intervals
            for _ in range(30):  # 30 x 1 second = 30 seconds total
                if not self.running:
                    return
                time.sleep(1)
            
            if self.running:
                self._print_summary()
    
    def stop_collection(self):
        """Stop data collection"""
        self.logger.info("Stopping data collection...")
        self.running = False

        # FIX #10: Wait for threads to finish (with timeout)
        threads = [
            ('flush', self.flush_thread),
            ('summary', self.summary_thread),
            ('reconnection', self.reconnection_thread)
        ]
        
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                self.logger.info(f"Waiting for {name} thread to finish...")
                thread.join(timeout=5)
                if thread.is_alive():
                    self.logger.warning(f"{name} thread did not finish in time")

        # Unsubscribe from all feeds
        for sub_id in self.subscription_ids:
            try:
                pass  # self.info.unsubscribe(...) if available
            except Exception as e:
                self.logger.error(f"Error unsubscribing {sub_id}: {e}")

        # Use try/finally to ensure writers are closed even if flush fails
        try:
            self.logger.info("Performing final flush...")
            self._flush_buffers()
        except Exception as e:
            self.logger.error(f"Error during final flush: {e}")
        finally:
            # ALWAYS close all writers
            self.logger.info("Closing all parquet writers...")
            with self.writer_lock:
                for key in list(self.active_writers.keys()):
                    self._close_writer(key[0], key[1])

        # Disconnect websocket
        try:
            if self.info:
                self.info.disconnect_websocket()
        except Exception as e:
            self.logger.error(f"Error disconnecting websocket: {e}")

        # Final summary
        self._print_summary()
        self.logger.info(f"Data files saved in: {self.output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function"""
    # Configuration
    SYMBOLS = ["BTC", "WLFI", "PAXG"]
    OUTPUT_DIR = "HL_data"
    ORDERBOOK_DEPTH = 20
    
    print("Hyperliquid Tick Data Collector (Fixed Version)")
    print("=" * 50)
    print(f"Symbols: {SYMBOLS}")
    print(f"Order book depth: {ORDERBOOK_DEPTH} levels")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    collector = HyperliquidDataCollector(
        SYMBOLS,
        OUTPUT_DIR,
        orderbook_depth=ORDERBOOK_DEPTH
    )
    
    collector.start_collection()


if __name__ == "__main__":
    main()
