# Advanced Avellaneda-Stoikov Market Making with Freqtrade

A sophisticated market making system built on Freqtrade, implementing the Avellaneda-Stoikov optimal market making model with real-time parameter calculation for dynamic spread optimization. Runs on Hyperliquid. It is Long-Only and Ping-Pong for now.

⚠️: Data in `HL_data_collector/HL_data` is a subsample of the real BTC/USDC Perp data on Hyperliquid. Due to GitHub limitations, I cannot include the actual data, which is approximately a hundred MB per day. You should collect data yourself for several days before obtaining reliable {`σ`, `κ`, `γ`} parameter estimation/calibration. Running `docker-compose build` then `docker-compose up` will start data collection as well as Freqtrade trading with inaccurate {`σ`, `κ`, `γ`} parameters for the first couple of days. After running for a couple of days, you can leave it as is or reset the Freqtrade trading by stopping with `docker-compose down`, removing the `tradesv3.sqlite` database, and restarting with `docker-compose up` (`docker-compose up -d` to run as a daemon in the background). The code updates the {`σ`, `κ`, `γ`} parameters used for live trading every day, once a day.

## Overview

This project implements an advanced market making strategy that:

- **Dynamically calculates optimal bid-ask spreads** using the Avellaneda-Stoikov market making model.
- **Continuously adapts to market conditions** through real-time parameter estimation (`σ`, `κ`, `γ`). Once a day, parameters calculated from data of day N-1 (for `γ`) and N-2 (for `σ`, `κ`) are used for trading on day N. Therefore you need at least 2 days of data collection.
- **Uses Freqtrade**
- **Uses Hyperliquid exchange**

## Key Features

### 🎯 Dynamic Spread Calculation
- **Sigma parameter** (`σ`): Price volatility estimate
- **κ parameter** (`κ`): Order flow intensity factor
- **Gamma parameter** (`γ`): Risk aversion coefficient
- **Real-time recalibration** through automated parameter calculation. Updates once a day, parameters calculated from data of day N-1 (for `γ`) and N-2 (for `σ`, `κ`) are used for trading on day N. Therefore you need at least 2 days of data collection.

### 📊 Market Data Integration
- **Order book analysis** for mid-price calculation and spread optimization
- **Trade flow analysis** for order arrival intensity estimation
- **Volatility estimation** from recent price movements

### 🏗️ Modular Architecture
- **Core strategy**: `avellaneda.py` - Main Avellaneda-Stoikov Freqtrade strategy implementation
- **Parameter calculation**: `calculate_avellaneda_parameters.py` - Unified estimation of parameters {`σ`, `κ`, `γ`}
- **Data collection**: `HL_data_collector/` - Separate service for market data gathering (time-tagged bid/ask spread, time-tagged filled order list, time-tagged orderbooks)
- **Parameter runner**: `run_avellaneda_param_calculation.py` - Automated {`σ`, `κ`, `γ`} parameters updates once a day

## Project Structure

```
ADVANCED_MM/
├── user_data/
│   ├── strategies/
│   │   ├── avellaneda.py             # Main Avellaneda-Stoikov strategy
│   │   ├── run_avellaneda_param_calculation.py # Parameter calculation runner
│   ├── config.json                   # Freqtrade configuration (BTC/USDC)
│   └── [other standard freqtrade dirs] # backtest_results/, data/, logs/, etc.
├── scripts/
│   ├── calculate_avellaneda_parameters.py # Unified parameter calculation
│   ├── avellaneda_parameters_BTC.json # Current model parameters (BTC)
│   ├── Francesco_Mangia_Avellaneda_BTC.ipynb # Research notebook describing most of what was implemented, with some changes
│   └── requirements.txt              # Python dependencies
├── HL_data_collector/                # Separate data collection service
│   ├── hyperliquid_data_collector.py # Market data gathering
│   ├── run_collector.py              # Data collector orchestrator
│   ├── HL_data/                      # Folder containing collected market data (CSV files)
│   ├── Dockerfile                    # Data collector docker container build
│   └── requirements.txt              # Python dependencies
├── docker-compose.yml                # Main container orchestration
├── Dockerfile.technical              # Extra python libraries installation for docker compose
└── show_PnL.py                       # Profit and loss analysis display tool
```

## Mathematical Foundation

### Avellaneda-Stoikov Market Making Model

The strategy implements the classical Avellaneda-Stoikov optimal market making model from "High-frequency trading in a limit order book" (2008):

**Core Model Elements:**

### Avellaneda & Stoikov's Paper

Avellaneda and Stoikov's paper study the optimal submission strategies of bid and ask orders in a limit order book, "*balancing between the dealer’s personal risk considerations and the market environment*" and defining the bid/ask spread as:

$$
\text{bid / ask spread} = \text{spread} = \gamma \sigma ^2 (T - t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{κ}\right)
$$

This spread is defined around a reservation price i.e. a price at which a market maker is indifferent between their current portfolio and their current portfolio $\pm$ a new share. The reservation price is derived in the whitepaper as follows:

$$
\text{reservation price} = r = s - q\gamma\sigma^2(T-t)
$$

$$
\text{gap} = |r - s|
$$

Where:

* $s$ the mid-price of the asset
* $\sigma$, the volatility of the asset
* $κ$, the intensity of the arrival of orders
* $\gamma$, a risk factor that is adjusted with the best backtest (maximum sharpe ratio)
* $T$, the end of the time series, (T - t) here is between 0 and 1 and is the fractional number of days left before the end of day.
* $q$, the number of assets held in inventory

And the final best buy and sell limit order prices:

$$
\begin{aligned}
\text{buy}_{\text{price}} &= r - \delta_b \\
\text{sell}_{\text{price}} &= r + \delta_a
\end{aligned}
$$

If $r \geq s$:

$$
\begin{aligned}
\delta_a &= \text{spread}/2 + \text{gap} \\
\delta_b &= \text{spread}/2 - \text{gap}
\end{aligned}
$$

If $r < s$:

$$
\begin{aligned}
\delta_a &= \text{spread}/2 - \text{gap} \\
\delta_b &= \text{spread}/2 + \text{gap}
\end{aligned}
$$


### Parameter Estimation and Calibration

**Gamma (γ) - Risk Aversion:**
- Controls the trade-off between profit and inventory risk
- Higher γ → tighter spreads around zero inventory
- Estimated with best daily backtest, i.e. with maximum sharpe ratio

**Kappa (κ) Parameter - Order Flow Intensity:**
- Estimated from order book data: `λ(δ) = A e^(-κ δ)`
- Fitted using historical time-tagged filled order list and bid-ask spreads

**Sigma (σ) - Volatility:**
- Estimated from recent prices using rolling windows

There is one {`σ`, `κ`, `γ`} set calculated and used per day. The set calculated using day N-1 data is used for day N.

## Key Components

### avellaneda.py

The main Freqtrade strategy implementing the Avellaneda-Stoikov model:
- **Dynamic spread calculation** using {`σ`, `κ`, `γ`} parameters
- **Inventory-aware pricing** with asymmetric bid-ask spreads
- **Real-time parameter loading** from `avellaneda_parameters_BTC.json`
- **Automated parameter recalculation** through integrated runner

### Parameter Calculation Scripts

- **calculate_avellaneda_parameters.py**: Unified parameter estimation for {`σ`, `κ`, `γ`}
- **run_avellaneda_param_calculation.py**: Strategy-integrated regular parameter updates (daily)
- **Reference notebook**: Francesco_Mangia_Avellaneda_BTC.ipynb used as a basis, but several changes

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in Dry-Run (paper trading) mode before deploying with real capital. Past performance does not guarantee future results.

## License

This project implements academic market making models and is intended for research and educational use.











































































