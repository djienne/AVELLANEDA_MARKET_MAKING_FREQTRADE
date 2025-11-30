# Hyperliquid Data Collector

Real-time tick data collector for Hyperliquid cryptocurrency exchange using websockets (order books, trades, bid/ask quotes) with automatic reconnection on connection drops.

## Overview

`hyperliquid_data_collector.py` connects to Hyperliquid's websocket API to collect and store:
- **Price data** (best bid/offer)
- **Trade executions**
- **Order book snapshots** (configurable depth, default: 20 levels)

The collector includes robust connection handling with automatic reconnection when websocket connections are dropped or interrupted.

## Dependencies

The project's dependencies are listed in `requirements.txt`.

To install the core dependencies for running the data collector, you can use:
```bash
pip install hyperliquid-python-sdk websocket-client pyarrow pandas
```

The `requirements.txt` file also contains libraries for data analysis (`numpy`, `scipy`, etc.), which are used by the parameter calculation scripts.

## Usage

The recommended way to run the data collector is using the `run_collector.py` script, which allows for configuration via environment variables.

```bash
export SYMBOLS="BTC,ETH,SOL"
export OUTPUT_DIR="HL_data"
export ORDERBOOK_DEPTH=20

python run_collector.py
```

### Configuration

- `SYMBOLS`: Comma-separated list of symbols to collect data for (e.g., "BTC,ETH,PAXG"). Defaults to `"BTC,WLFI,PAXG"`.
- `OUTPUT_DIR`: Directory to store the output files. Defaults to `"HL_data"`.
- `ORDERBOOK_DEPTH`: The number of order book levels to store. Defaults to `20`.

Alternatively, you can run the `hyperliquid_data_collector.py` script directly, but the configuration is hardcoded within the script's `main` function.

## Docker

The project includes a `Dockerfile` to run the data collector in a container.

**Build the Docker image:**
```bash
docker build -t hyperliquid-collector .
```

**Run the Docker container:**
```bash
docker run -d --name hl-collector \
  -e SYMBOLS="BTC,ETH,SOL,WLFI" \
  -v ./HL_data:/app/HL_data \
  hyperliquid-collector
```
This will start the collector in the background, save data to the local `HL_data` directory, and use the specified symbols.

## Data Storage

Data is written to the output directory (default: `HL_data/`) in **Parquet** format. Each data type for each symbol is stored in its own directory containing partitioned parquet files.

### File Structure
```
HL_data/
├── prices_{SYMBOL}.parquet/     # Directory for Best bid/ask prices
│   └── part_{timestamp}.parquet
├── trades_{SYMBOL}.parquet/     # Directory for Trade executions
│   └── part_{timestamp}.parquet
└── orderbooks_{SYMBOL}.parquet/ # Directory for Order book snapshots
    └── part_{timestamp}.parquet
```

### Data Schema

**prices_{SYMBOL}.parquet**
- `timestamp`: Collection timestamp (float)
- `symbol`: Trading symbol (string)
- `price`: Bid/ask price (float)
- `size`: Volume at price level (float)
- `side`: "bid" or "ask" (string)
- `exchange_timestamp`: Hyperliquid timestamp (int, optional)

**trades_{SYMBOL}.parquet**
- `timestamp`: Collection timestamp (float)
- `symbol`: Trading symbol (string)
- `price`: Trade price (float)
- `size`: Trade volume (float)
- `side`: "buy" or "sell" (string)
- `trade_id`: Unique trade identifier (string, optional)
- `exchange_timestamp`: Hyperliquid timestamp (int, optional)

**orderbooks_{SYMBOL}.parquet**
- `timestamp`: Collection timestamp (float)
- `symbol`: Trading symbol (string)
- `sequence`: Order book sequence number (int, optional)
- `exchange_timestamp`: Hyperliquid timestamp (int, optional)
- `bid_price_{i}`, `bid_size_{i}`: Bid levels 0-19 (float)
- `ask_price_{i}`, `ask_size_{i}`: Ask levels 0-19 (float)

## Avellaneda-Stoikov Parameter Calculation

*Note: The script `calculate_avellaneda_parameters.py` is currently pending implementation.*

This script will calculate optimal market making parameters using the Avellaneda-Stoikov model. It analyzes historical trade and price data to estimate volatility (sigma) and order arrival intensity (A, k). It then backtests different risk aversion (gamma) values to find the optimal parameter that maximizes profitability while managing inventory risk.

`avellaneda_parameters_BTC.json`: This file stores the output of the calculation for a specific ticker (e.g., BTC). It includes key market data (mid-price, sigma, A, k), the optimized gamma, and the resulting reservation price and optimal bid/ask quotes. This provides a snapshot of the recommended market making parameters.
