# Advanced Avellaneda-Stoikov Market Making with Freqtrade

A sophisticated market making system built on Freqtrade, implementing the Avellaneda-Stoikov optimal market making model with real-time parameter calculation for dynamic spread optimization. Runs on Hyperliquid. It is long-only and ping-pong for now.

## Important Note on Data Collection

Reliable sigma (volatility), k (order flow intensity), and gamma (risk aversion) parameters are crucial for this strategy. The included data in `HL_data_collector/HL_data` is only a small sample. You **must** collect your own data for at least a few days to obtain accurate parameter estimations.

The system is designed to be self-sufficient:
1. Run `docker-compose build` and `docker-compose up` to start both data collection and trading.
2. Initially, the trading bot will use inaccurate parameters.
3. The Avellaneda parameter sets are automatically recalculated **at most once every 15 minutes** (rate-limited). The strategy triggers recalculation every 10 bot loops (~150 seconds with a 15 seconds `process_throttle_secs`), and a lock file prevents running more frequently than once per 15 minutes. After a couple of days, the parameters will become more reliable.
4. You can either let the system run continuously or, for a fresh start with better parameters, stop the services (`docker-compose down`), delete the `tradesv3.sqlite` database, and restart (`docker-compose up`).

## Overview

This project implements an advanced market making strategy for Hyperliquid, dynamically calculating optimal bid-ask spreads using the Avellaneda-Stoikov model with real-time parameter estimation.

**Current Configuration:** The included configuration is set to trade PAXG/USDC. The system includes pre-calculated parameters for two trading pairs: PAXG and ETH. To switch trading pairs, change `exchange.pair_whitelist` in `user_data/config.json`; the strategy and parameter tooling will pick up the first pair and look for `avellaneda_parameters_{TICKER}.json` automatically.

## Project Structure

```
ADVANCED_MM/
|-- user_data/
|   |-- strategies/
|   |   |-- avellaneda.py             # Main Avellaneda-Stoikov strategy
|   |   `-- run_avellaneda_param_calculation.py # Parameter calculation runner
|   |-- config.json                   # Freqtrade configuration
|   `-- [other standard freqtrade dirs] # backtest_results/, data/, logs/, etc.
|-- scripts/
|   |-- calculate_avellaneda_parameters.py # Unified parameter calculation
|   |-- avellaneda_parameters_PAXG.json # Pre-calculated parameters for PAXG
|   |-- avellaneda_parameters_ETH.json  # Pre-calculated parameters for ETH
|   |-- backtest.py                   # Backtesting for parameter optimization
|   |-- volatility.py                 # Volatility calculation (GARCH model)
|   |-- intensity.py                  # Order flow intensity estimation (MLE)
|   |-- Francesco_Mangia_Avellaneda_BTC.ipynb # Research notebook
|   `-- requirements.txt              # Python dependencies
|-- HL_data_collector/
|   |-- hyperliquid_data_collector.py # Market data gathering
|   |-- run_collector.py              # Data collector orchestrator
|   |-- HL_data/                      # Folder containing collected market data
|   |-- Dockerfile                    # Data collector docker container build
|   `-- requirements.txt              # Python dependencies
|-- docker-compose.yml                # Main container orchestration
|-- Dockerfile.technical              # Extra python libraries for docker
`-- show_PnL.py                       # Profit and loss analysis display tool
```

## Building and Running

The project uses Docker and Docker Compose for containerization and orchestration.

* Build the Docker images: `docker-compose build`
* Start the trading bot and data collector: `docker-compose up`

The `docker-compose.yml` file defines two services:

* `freqtrade_mm`: This service runs the trading bot. It uses the `freqtradeorg/freqtrade:2025.7` image and mounts the `user_data`, `scripts`, and `HL_data_collector` directories. The command shows that it runs the `avellaneda` strategy with the configuration from `user_data/config.json`.
* `hl-collector`: This service runs the data collector. It builds a Docker image from `HL_data_collector/Dockerfile` and runs the `run_collector.py` script.

## Configuration

The main configuration for the Freqtrade bot is in the `user_data/config.json` file. Here are some of the key settings:

* "max_open_trades": 1 - The bot will only have one open trade at a time.
* "stake_currency": "USDC" - The currency used for trading.
* "stake_amount": 50 - The amount of stake currency to use for each trade.
* "dry_run": true - The bot is running in simulation mode.
* "trading_mode": "futures" - The bot is trading futures contracts.
* "exchange.name": "hyperliquid" - The exchange to trade on.
* "exchange.pair_whitelist": ["PAXG/USDC:USDC"] - The trading pair to use.

## Switching Trading Pairs

Change the pair once in `user_data/config.json` under `exchange.pair_whitelist` (first entry) and the rest will follow. The strategy reads that pair to pick the matching parameter file `avellaneda_parameters_{TICKER}.json`, and the parameter calculator defaults to the same ticker when no CLI ticker is provided.

Parameter files live in `scripts/` by default (override with `AVELLANEDA_PARAMS_DIR`). After editing the pair, regenerate parameters with:
```
python scripts/calculate_avellaneda_parameters.py
# or override explicitly
python scripts/calculate_avellaneda_parameters.py ETH
```
The data collector is assumed to include this pair (and potentially more); adjust its env vars only if you add a pair it doesn't already track.

## Mathematical Foundation

### Avellaneda-Stoikov Market Making Model

The strategy implements the classical Avellaneda-Stoikov optimal market making model from "High-frequency trading in a limit order book" (2008).

**Core Model Elements:**

The bid/ask spread is defined as:

```
spread = gamma * sigma**2 * T + (2 / gamma) * ln(1 + gamma / k)
```

This spread is centered around a reservation price `r`, which is the price at which a market maker is indifferent to buying or selling another share.

```
reservation price r = s - q * gamma * sigma**2 * T

gap = |r - s|
```

Where:
- `s`: mid-price of the asset
- `sigma`: volatility of the asset
- `k`: intensity of the arrival of orders
- `gamma`: risk factor, optimized via backtests over the entire historical data, alongside the time horizon
- `T`: optimized fixed time horizon (in hours), representing the urgency to liquidate inventory
- `q`: number of assets held in inventory (forced to 0 for neutral pricing)

And the final best buy and sell limit order prices:

```
buy_price = r - delta_b
sell_price = r + delta_a
```

If r >= s:

```
delta_a = spread/2 + gap
delta_b = spread/2 - gap
```

If r < s:

```
delta_a = spread/2 - gap
delta_b = spread/2 + gap
```

### Parameter Estimation

Parameters are recalculated automatically, rate-limited to **at most once every 15 minutes**:

**Recalculation Mechanism:**
- The strategy attempts to trigger recalculation every 10 bot loops (~150 seconds with 15-seconds `process_throttle_secs`).
- A lock file (`.avellaneda_last_run.json`) prevents execution if parameters were calculated within the last 15 minutes.
- This ensures parameters update regularly while preventing excessive computation.

**Parameter Calculations:**
- **gamma (Risk Aversion):** Optimized via backtests over the entire historical data, alongside the time horizon.
- **T (Time Horizon):** Optimized via backtests over the entire historical data, alongside the risk aversion parameter.
- **k (Order Flow Intensity):** Estimated using Maximum Likelihood Estimation (MLE) on raw fill counts for robustness.
- **sigma (Volatility):** Calculated using a GARCH(1,1) model to capture volatility clustering. A rolling window standard deviation of price movements is used as a fallback if the GARCH model fails or if there is insufficient data.

## Disclaimer

This software is for educational and research purposes only. Market making involves significant financial risk. Always test thoroughly in Dry-Run (paper trading) mode before deploying with real capital. Past performance does not guarantee future results.

## License

This project implements academic market making models and is intended for research and educational use.
