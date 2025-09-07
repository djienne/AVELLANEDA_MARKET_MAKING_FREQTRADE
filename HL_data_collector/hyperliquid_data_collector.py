#!/usr/bin/env python3
"""
Hyperliquid Tick Data Collector
Collects time-tagged prices, executed orders, and order book data via websockets
"""

import asyncio
import json
import time
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import random
import logging

from hyperliquid.info import Info
from hyperliquid.utils import constants


@dataclass
class TickData:
    """Base class for all tick data"""
    timestamp: float
    symbol: str
    exchange_timestamp: Optional[int] = None


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


class DataStats:
    """Track data collection statistics"""
    def __init__(self):
        self.start_time = time.time()
        self.counters = defaultdict(int)
        self.last_update = time.time()
        self.recent_data = defaultdict(lambda: deque(maxlen=100))
        self.reconnection_count = 0
        self.last_data_time = time.time()
    
    def update(self, data_type: str, data: Any = None):
        self.counters[data_type] += 1
        self.last_update = time.time()
        self.last_data_time = time.time()
        if data:
            self.recent_data[data_type].append(data)
    
    def record_reconnection(self):
        self.reconnection_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
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


class HyperliquidDataCollector:
    """Main data collector class"""
    
    def __init__(self, symbols: List[str], output_dir: str = "data", orderbook_depth: int = 20):
        self.symbols = symbols
        self.output_dir = output_dir
        self.orderbook_depth = orderbook_depth  # Configurable order book depth
        self.info = None
        self.stats = DataStats()
        self.subscription_ids = []
        self.connection_healthy = False
        self.reconnection_delay = 1  # Start with 1 second delay
        self.max_reconnection_delay = 300  # Max 5 minutes
        self.data_timeout = 60  # Consider connection dead if no data for 60 seconds
        
        # Create separate buffers for each symbol
        self.data_buffers = {}
        for symbol in symbols:
            self.data_buffers[symbol] = {
                'prices': deque(maxlen=10000),
                'trades': deque(maxlen=10000),
                'orderbooks': deque(maxlen=1000)
            }
        
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.reconnection_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # File paths for each symbol
        self.symbol_files = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Initialize connection
        self._init_connection()
    
    def _init_csv_files(self):
        """Initialize CSV files for data storage - separate files per symbol"""
        
        for symbol in self.symbols:
            # Initialize file paths for this symbol
            self.symbol_files[symbol] = {}
            
            # Price data CSV for this symbol
            price_file = os.path.join(self.output_dir, f"prices_{symbol}.csv")
            self.symbol_files[symbol]['prices'] = price_file
            # Only write header if file doesn't exist
            if not os.path.exists(price_file):
                with open(price_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'price', 'size', 'side', 'exchange_timestamp'])
            
            # Trade data CSV for this symbol
            trade_file = os.path.join(self.output_dir, f"trades_{symbol}.csv")
            self.symbol_files[symbol]['trades'] = trade_file
            # Only write header if file doesn't exist
            if not os.path.exists(trade_file):
                with open(trade_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'price', 'size', 'side', 'trade_id', 'exchange_timestamp'])
            
            # Order book CSV for this symbol (configurable depth)
            orderbook_file = os.path.join(self.output_dir, f"orderbooks_{symbol}.csv")
            self.symbol_files[symbol]['orderbooks'] = orderbook_file
            # Only write header if file doesn't exist
            if not os.path.exists(orderbook_file):
                with open(orderbook_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = ['timestamp', 'sequence', 'exchange_timestamp']
                    for i in range(self.orderbook_depth):
                        headers.extend([f'bid_price_{i}', f'bid_size_{i}', f'ask_price_{i}', f'ask_size_{i}'])
                    writer.writerow(headers)
    
    def _init_connection(self):
        """Initialize websocket connection"""
        try:
            self.info = Info(constants.MAINNET_API_URL, skip_ws=False)
            self.connection_healthy = True
            logging.info("Websocket connection initialized")
        except Exception as e:
            logging.error(f"Failed to initialize connection: {e}")
            self.connection_healthy = False
    
    def _is_connection_healthy(self) -> bool:
        """Check if connection is healthy based on recent data"""
        if not self.connection_healthy:
            return False
        
        time_since_last_data = time.time() - self.stats.last_data_time
        if time_since_last_data > self.data_timeout:
            logging.warning(f"No data received for {time_since_last_data:.1f} seconds - connection may be dead")
            return False
        
        return True
    
    def _reconnect(self):
        """Reconnect to websocket with exponential backoff"""
        logging.info(f"Attempting reconnection (delay: {self.reconnection_delay}s)")
        
        # Wait before reconnecting
        time.sleep(self.reconnection_delay)
        
        try:
            # Cleanup old connection
            if self.info:
                try:
                    self.info.disconnect_websocket()
                except:
                    pass
            
            # Create new connection
            self._init_connection()
            
            if self.connection_healthy:
                # Resubscribe to all feeds
                self.subscription_ids.clear()
                self._subscribe_to_feeds()
                
                # Reset reconnection delay on successful connection
                self.reconnection_delay = 1
                self.stats.record_reconnection()
                logging.info("Successfully reconnected and resubscribed to feeds")
                return True
            
        except Exception as e:
            logging.error(f"Reconnection failed: {e}")
        
        # Increase delay for next attempt (exponential backoff)
        self.reconnection_delay = min(self.reconnection_delay * 2 + random.uniform(0, 1), self.max_reconnection_delay)
        return False
    
    def _subscribe_to_feeds(self):
        """Subscribe to all data feeds"""
        for symbol in self.symbols:
            logging.info(f"Subscribing to data feeds for {symbol}...")
            
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
        
        logging.info(f"Subscribed to {len(self.subscription_ids)} data feeds")
    
    def _connection_monitor(self):
        """Monitor connection health and reconnect if needed"""
        while self.running:
            time.sleep(10)  # Check every 10 seconds
            
            if not self.running:
                break
                
            if not self._is_connection_healthy():
                self.connection_healthy = False
                logging.warning("Connection unhealthy - attempting reconnection")
                
                # Try to reconnect
                while self.running and not self._reconnect():
                    if not self.running:
                        break
    
    def _write_to_csv(self, filename: str, data: List[Any]):
        """Write data to CSV file"""
        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                for item in data:
                    if isinstance(item, dict):
                        writer.writerow(item.values())
                    else:
                        writer.writerow(asdict(item).values())
        except Exception as e:
            print(f"Error writing to CSV {filename}: {e}")
    
    def _handle_bbo_data(self, data: Dict[str, Any]):
        """Handle best bid/offer data"""
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'bbo' and 'data' in data:
                bbo_data = data['data']
                symbol = bbo_data.get('coin', 'UNKNOWN')
                
                # BBO format: bbo array with [bid, ask]
                if 'bbo' in bbo_data and len(bbo_data['bbo']) >= 2:
                    bid_info = bbo_data['bbo'][0]  # First element is bid
                    ask_info = bbo_data['bbo'][1]  # Second element is ask
                    
                    # Create bid data
                    bid_data = {
                        'timestamp': timestamp,
                        'price': float(bid_info['px']),
                        'size': float(bid_info['sz']),
                        'side': 'bid',
                        'exchange_timestamp': bbo_data.get('time')
                    }
                    self.data_buffers[symbol]['prices'].append(bid_data)
                    
                    # Create ask data
                    ask_data = {
                        'timestamp': timestamp,
                        'price': float(ask_info['px']),
                        'size': float(ask_info['sz']),
                        'side': 'ask',
                        'exchange_timestamp': bbo_data.get('time')
                    }
                    self.data_buffers[symbol]['prices'].append(ask_data)
                
                self.stats.update('bbo_updates')
            else:
                # Direct format fallback
                symbol = data.get('coin', 'UNKNOWN')
                
                if 'bid' in data and data['bid']:
                    bid_data = {
                        'timestamp': timestamp,
                        'price': float(data['bid']['px']),
                        'size': float(data['bid']['sz']),
                        'side': 'bid',
                        'exchange_timestamp': data.get('time')
                    }
                    self.data_buffers[symbol]['prices'].append(bid_data)
                
                if 'ask' in data and data['ask']:
                    ask_data = {
                        'timestamp': timestamp,
                        'price': float(data['ask']['px']),
                        'size': float(data['ask']['sz']),
                        'side': 'ask',
                        'exchange_timestamp': data.get('time')
                    }
                    self.data_buffers[symbol]['prices'].append(ask_data)
                
                self.stats.update('bbo_updates')
        except Exception as e:
            print(f"Error handling BBO data: {e}")
    
    def _handle_trade_data(self, data: Dict[str, Any]):
        """Handle trade data"""
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'trades' and 'data' in data:
                trades = data['data']
                for trade in trades:
                    symbol = trade.get('coin', 'UNKNOWN')
                    # Convert A/B to buy/sell
                    side = 'sell' if trade['side'] == 'A' else 'buy'
                    
                    trade_data = {
                        'timestamp': timestamp,
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': side,
                        'trade_id': str(trade.get('tid')),
                        'exchange_timestamp': trade.get('time')
                    }
                    self.data_buffers[symbol]['trades'].append(trade_data)
                
                self.stats.update('trades', len(trades))
            else:
                # Direct format fallback  
                if isinstance(data, list):
                    trades = data
                else:
                    trades = [data]
                    
                for trade in trades:
                    symbol = trade.get('coin', 'UNKNOWN')
                    side = 'sell' if trade['side'] == 'A' else 'buy'
                    
                    trade_data = {
                        'timestamp': timestamp,
                        'price': float(trade['px']),
                        'size': float(trade['sz']),
                        'side': side,
                        'trade_id': str(trade.get('tid')),
                        'exchange_timestamp': trade.get('time')
                    }
                    self.data_buffers[symbol]['trades'].append(trade_data)
                
                self.stats.update('trades', len(trades))
        except Exception as e:
            print(f"Error handling trade data: {e}")
    
    def _handle_orderbook_data(self, data: Dict[str, Any]):
        """Handle order book data"""
        try:
            timestamp = time.time()
            
            # Handle channel-based format
            if 'channel' in data and data['channel'] == 'l2Book' and 'data' in data:
                book_data = data['data']
                symbol = book_data.get('coin', 'UNKNOWN')
                
                # Parse bids and asks
                bids = []
                asks = []
                
                if 'levels' in book_data and len(book_data['levels']) >= 2:
                    # levels[0] is bids array, levels[1] is asks array
                    bids_array = book_data['levels'][0]
                    asks_array = book_data['levels'][1]
                    
                    # Parse bids
                    for bid in bids_array:
                        bids.append(OrderBookLevel(price=float(bid['px']), size=float(bid['sz'])))
                    
                    # Parse asks  
                    for ask in asks_array:
                        asks.append(OrderBookLevel(price=float(ask['px']), size=float(ask['sz'])))
                
                # Prepare data for CSV (flatten configurable depth levels)
                csv_row = {
                    'timestamp': timestamp,
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
                
                self.data_buffers[symbol]['orderbooks'].append(csv_row)
                self.stats.update('orderbook_updates')
            else:
                # Direct format fallback
                symbol = data.get('coin', 'UNKNOWN')
                
                # Parse bids and asks
                bids = []
                asks = []
                
                if 'levels' in data and len(data['levels']) >= 2:
                    # levels[0] is bids array, levels[1] is asks array
                    bids_array = data['levels'][0]
                    asks_array = data['levels'][1]
                    
                    # Parse bids
                    for bid in bids_array:
                        bids.append(OrderBookLevel(price=float(bid['px']), size=float(bid['sz'])))
                    
                    # Parse asks  
                    for ask in asks_array:
                        asks.append(OrderBookLevel(price=float(ask['px']), size=float(ask['sz'])))
                
                # Prepare data for CSV (flatten configurable depth levels)
                csv_row = {
                    'timestamp': timestamp,
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
                
                self.data_buffers[symbol]['orderbooks'].append(csv_row)
                self.stats.update('orderbook_updates')
            
        except Exception as e:
            print(f"Error handling order book data: {e}")
    
    def _flush_buffers(self):
        """Flush data buffers to CSV files - separate files per symbol"""
        try:
            for symbol in self.symbols:
                symbol_buffers = self.data_buffers[symbol]
                symbol_files = self.symbol_files[symbol]
                
                # Flush prices for this symbol
                if symbol_buffers['prices']:
                    prices_to_write = list(symbol_buffers['prices'])
                    symbol_buffers['prices'].clear()
                    self.executor.submit(self._write_to_csv, symbol_files['prices'], prices_to_write)
                
                # Flush trades for this symbol
                if symbol_buffers['trades']:
                    trades_to_write = list(symbol_buffers['trades'])
                    symbol_buffers['trades'].clear()
                    self.executor.submit(self._write_to_csv, symbol_files['trades'], trades_to_write)
                
                # Flush order books for this symbol
                if symbol_buffers['orderbooks']:
                    orderbooks_to_write = list(symbol_buffers['orderbooks'])
                    symbol_buffers['orderbooks'].clear()
                    self.executor.submit(self._write_to_csv, symbol_files['orderbooks'], orderbooks_to_write)
                
        except Exception as e:
            print(f"Error flushing buffers: {e}")
    
    def _print_summary(self):
        """Print data collection summary"""
        summary = self.stats.get_summary()
        print("\n" + "="*60)
        print(f"DATA COLLECTION SUMMARY - {summary['last_update']}")
        print("="*60)
        print(f"Runtime: {summary['runtime_formatted']}")
        print(f"Connection: {'🟢 Healthy' if self.connection_healthy else '🔴 Reconnecting'}")
        if summary['reconnections'] > 0:
            print(f"Reconnections: {summary['reconnections']}")
        print(f"Time since last data: {summary['seconds_since_last_data']:.1f}s")
        print(f"Data collected:")
        for data_type, count in summary['counters'].items():
            rate = summary['rates_per_minute'].get(data_type, 0)
            print(f"  {data_type}: {count:,} ({rate:.1f}/min)")
        
        print(f"\nBuffer sizes by symbol:")
        for symbol in self.symbols:
            symbol_buffers = self.data_buffers[symbol]
            total_buffered = sum(len(buffer) for buffer in symbol_buffers.values())
            print(f"  {symbol}: {total_buffered} ({len(symbol_buffers['prices'])} prices, {len(symbol_buffers['trades'])} trades, {len(symbol_buffers['orderbooks'])} orderbooks)")
        
        print("="*60)
    
    def start_collection(self):
        """Start data collection"""
        print(f"Starting Hyperliquid data collection for symbols: {self.symbols}")
        print(f"Output directory: {self.output_dir}")
        
        self.running = True
        
        try:
            # Subscribe to data feeds
            if self.connection_healthy:
                self._subscribe_to_feeds()
            else:
                logging.error("Initial connection failed - will attempt reconnection")
            
            # Start background tasks
            flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
            flush_thread.start()
            
            summary_thread = threading.Thread(target=self._periodic_summary, daemon=True)
            summary_thread.start()
            
            # Start connection monitoring thread
            self.reconnection_thread = threading.Thread(target=self._connection_monitor, daemon=True)
            self.reconnection_thread.start()
            
            # Keep running
            print("Data collection started. Press Ctrl+C to stop.")
            print("Automatic reconnection enabled - connection drops will be handled automatically.")
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop_collection()
        except Exception as e:
            print(f"Error during data collection: {e}")
            self.stop_collection()
    
    def _periodic_flush(self):
        """Periodically flush buffers to disk"""
        while self.running:
            time.sleep(5)  # Flush every 5 seconds
            self._flush_buffers()
    
    def _periodic_summary(self):
        """Periodically print collection summary"""
        while self.running:
            time.sleep(30)  # Print summary every 30 seconds
            if self.running:
                self._print_summary()
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        
        # Unsubscribe from all feeds
        for sub_id in self.subscription_ids:
            try:
                # Note: The exact unsubscribe method depends on the SDK implementation
                pass  # self.info.unsubscribe(...) if available
            except Exception as e:
                print(f"Error unsubscribing {sub_id}: {e}")
        
        # Final flush
        self._flush_buffers()
        
        # Wait for executor to finish
        self.executor.shutdown(wait=True)
        
        # Disconnect websocket
        try:
            self.info.disconnect_websocket()
        except Exception as e:
            print(f"Error disconnecting websocket: {e}")
        
        # Final summary
        self._print_summary()
        print(f"\nData files saved in: {self.output_dir}")


def main():
    """Main function"""
    # Configuration
    SYMBOLS = ["BTC", "ETH", "SOL", "WLFI"]  # Add more symbols as needed
    OUTPUT_DIR = "HL_data"
    ORDERBOOK_DEPTH = 20  # Number of order book levels to capture (default: 20)
    
    print("Hyperliquid Tick Data Collector")
    print("================================")
    print(f"Order book depth: {ORDERBOOK_DEPTH} levels")
    
    # Create collector with configurable order book depth
    collector = HyperliquidDataCollector(SYMBOLS, OUTPUT_DIR, orderbook_depth=ORDERBOOK_DEPTH)
    
    # Start collection
    collector.start_collection()


if __name__ == "__main__":
    main()