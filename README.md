# Advanced Avellaneda-Stoikov Market Making with Freqtrade

A sophisticated market making system built on Freqtrade, implementing the Avellaneda-Stoikov optimal market making model with real-time parameter calculation for dynamic spread optimization. Runs on Hyperliquid. It is long-only and ping-pong for now.

## Important Note on Data Collection

Reliable sigma (volatility), k (order flow intensity), and gamma (risk aversion) parameters are crucial for this strategy. The included data in `HL_data_collector/HL_data` is only a small sample. You **must** collect your own data for at least a few days to obtain accurate parameter estimations.

The system is designed to be self-sufficient:
1. Run `docker-compose build` and `docker-compose up` to start data collection, parameter calculation and trading.
2. The `hl-params` service recalculates the parameters every 15 minutes from the last 24 h of collected data. It needs at least 45 minutes of data before it writes a first parameter file.
3. The bot opens new trades only while the parameter file is less than 1 hour old **and** the calculator measured a positive edge (see [Parameter Estimation](#parameter-estimation)). Otherwise it logs `New entries blocked` and only manages exits; dry-run behaves the same. Expect it to be blocked often: that is the honest answer when the data shows no edge.

## Overview

This project implements an advanced market making strategy for Hyperliquid, dynamically calculating optimal bid-ask spreads using the Avellaneda-Stoikov model with real-time parameter estimation.

**Current Configuration:** The included configuration is set to trade PAXG/USDC. Parameter files `avellaneda_parameters_{TICKER}.json` are generated at runtime (not committed). To switch trading pairs, change `exchange.pair_whitelist` in `user_data/config.json`; the strategy and the calculator pick up the first pair.

**💰 Support this project**:
- **Hyperliquid**: Sign up with [this referral link](https://app.hyperliquid.xyz/join/FREQTRADE) for 10% fee reduction

## Project Structure

```
ADVANCED_MM/
|-- user_data/
|   |-- strategies/
|   |   |-- avellaneda.py             # Main Avellaneda-Stoikov strategy
|   |   `-- pair_loader.py            # Active pair from config.json
|   |-- config.json                   # Freqtrade configuration
|   `-- [other standard freqtrade dirs] # backtest_results/, data/, logs/, etc.
|-- scripts/
|   |-- calculate_avellaneda_parameters.py # Unified parameter calculation
|   |-- backtest.py                   # Simulates the deployed bot to choose gamma
|   |-- volatility.py                 # Volatility calculation (GARCH model)
|   |-- intensity.py                  # Order flow intensity estimation (MLE)
|   |-- utils.py                      # Data loading, effective mid-price, tick size
|   |-- check_pipeline.py             # Runnable checks: python scripts/check_pipeline.py
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

* Create a `.env` file (gitignored) with the API-server credentials; `docker-compose` refuses to start the bot without it:
  ```
  FREQTRADE__API_SERVER__USERNAME=<user>
  FREQTRADE__API_SERVER__PASSWORD=<non-numeric password>
  FREQTRADE__API_SERVER__JWT_SECRET_KEY=<output of: python -c "import secrets;print(secrets.token_hex(32))">
  ```
  The previously committed secret is public in git history, so generate a new one. `show_PnL.py` reads the same two variables (e.g. `set -a; . ./.env; set +a; python show_PnL.py`).
* Build the Docker images: `docker-compose build`
* Start the trading bot and data collector: `docker-compose up`

The `docker-compose.yml` file defines three services:

* `freqtrade_mm`: the trading bot (`avellaneda` strategy, `user_data/config.json`). It only reads the parameter file.
* `hl-params`: runs `scripts/calculate_avellaneda_parameters.py` every 15 minutes, on the same image. Outside Docker, run the equivalent loop yourself:
  `while true; do python scripts/calculate_avellaneda_parameters.py; sleep 900; done`
* `hl-collector`: records Hyperliquid trades and order books to `HL_data_collector/HL_data` (files rotate every 5 minutes; set `SYMBOLS` to include your pair).

## Configuration

The main configuration for the Freqtrade bot is in the `user_data/config.json` file. Here are some of the key settings:

* "max_open_trades": 1 - The bot will only have one open trade at a time.
* "stake_currency": "USDC" - The currency used for trading.
* "stake_amount": 50 - The amount of stake currency to use for each trade.
* "dry_run": true - The bot is running in simulation mode.
* "fee": 0.0002 - Maker fee used for dry-run accounting. Same 2 bp as the simulation and the strategy (`MAKER_FEE` in `scripts/backtest.py`, passed to the bot through the parameter file).
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

The bot quotes each side at a distance from the mid-price `s`:

```
half_spread = 0.5 * gamma * (sigma * s)**2 * T + (1 / gamma) * ln(1 + gamma / k) + fee * s
buy_price   = s - half_spread(k_bid)
sell_price  = s + half_spread(k_ask)
```

Where:
- `s`: effective mid-price (price levels where $1000 of depth is reached on each side)
- `sigma`: daily volatility of log returns; `sigma * s` converts it to $
- `k`: decay of the fill intensity with distance from mid, `lambda(delta) = A * exp(-k * delta)`, in 1/$
- `gamma`: risk aversion, in 1/$
- `T`: time horizon in days, fixed to the 15-minute analysis window. Only `gamma * T` enters the first term, so `gamma` alone is tuned.
- The reservation price equals `s`: the inventory term `q * gamma * (sigma * s)**2 * T` is off (q = 0), because the bot is long-only with one position at a time (buy at the bid, then sell at the ask).

### Parameter Estimation

The `hl-params` service recalculates everything every 15 minutes from the last 24 h of data (50 chunks of 15 minutes). It refuses to run if the newest order book data is more than 10 minutes old (collector down); `--max-data-age` overrides this for offline analysis.

- **sigma:** GARCH(1,1) on 1-second log returns; a rolling standard deviation is used where GARCH fails or disagrees by more than 2x.
- **k, A:** Poisson maximum likelihood on how far taker trades walked from the mid-price. Each trade is compared with the book snapshot strictly before it, on the exchange clock.
- **gamma:** chosen by simulating the deployed bot on the recorded trades. The simulation holds one unit, long-only, alternating bid and ask, re-quoted every 15 s (`process_throttle_secs`) with the formula above. It uses the fee and the sigma and k from earlier periods only. An order fills when a taker trades strictly through it.
- **Edge and `trade_enabled`:** edge is the simulated PnL minus `mean(position) * price change`, which removes what a long-only bot earns or loses from the trend alone. `trade_enabled` is true only if the chosen gamma has a positive edge over the full window **and** the gamma chosen on the first 2/3 of the window keeps a positive edge on the last 1/3. The per-gamma fills and edges are saved in the parameter file under `backtest`.

**Limits to keep in mind**
- Freqtrade's dry run fills a limit order only when the top of the book crosses it at a loop instant. The simulation fills on any trade through the price, which is how the live exchange behaves. So dry-run PnL cannot validate the simulation.
- A few hours of data give few fills; `trade_enabled` is a noisy, deliberately conservative decision.
- When quotes sit far beyond the fitted distance range (49 ticks), k is extrapolated; the calculator prints a warning.

## Disclaimer

This software is for educational and research purposes only. Market making involves significant financial risk. Always test thoroughly in Dry-Run (paper trading) mode before deploying with real capital. Past performance does not guarantee future results.

## License

This project implements academic market making models and is intended for research and educational use.
