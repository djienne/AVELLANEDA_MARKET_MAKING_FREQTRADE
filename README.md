# Advanced Market Making with Freqtrade

A sophisticated market making system built on Freqtrade, implementing the Avellaneda-Stoikov optimal market making model with real-time parameter calculation for dynamic spread optimization. Runs on Hyperliquid.

⚠️: Data in `HL_data_collector/HL_data` is a subsample of the real data for BTC, because of Github limitations I cannot put realdata that is about hundred MB per day. You should collect yourself the data for several day before you can obtain a reliable parameter estimation/calibration.

## Overview

This project implements an advanced market making strategy that:

- **Dynamically calculates optimal bid-ask spreads** using the Avellaneda-Stoikov market making model
- **Continuously adapts to market conditions** through real-time parameter estimation (gamma, k, sigma)
- **Integrates with Hyperliquid exchange** for high-frequency trading
- **Uses mathematical optimization** to minimize inventory risk while maximizing profits

## Key Features

### 🎯 Dynamic Spread Calculation
- **Gamma parameter** (`γ`): Risk aversion coefficient controlling inventory penalties
- **K parameter** (`k`): Order flow intensity factor
- **Sigma parameter** (`σ`): Price volatility estimate
- **Real-time recalibration** through automated parameter calculation

### 📊 Market Data Integration
- **Order book analysis** for mid-price calculation and spread optimization
- **Trade flow analysis** for order arrival intensity estimation
- **Volatility estimation** from recent price movements

### 🔄 Automated Parameter Optimization
- **Real-time parameter calculation** using recent market data windows
- **Statistical analysis** of price movements and order flow
- **Automatic strategy recalibration** based on current market conditions

### 🏗️ Modular Architecture
- **Core strategy**: `avellaneda.py` - Main Avellaneda-Stoikov strategy implementation
- **Parameter calculation**: `calculate_avellaneda_parameters.py` - Unified parameter estimation
- **Data collection**: `HL_data_collector/` - Separate service for market data gathering
- **Parameter runner**: `run_avellaneda_param_calculation.py` - Automated parameter updates

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
│   ├── avellaneda_parameters_ETH.json # Current model parameters (ETH)
│   ├── Francesco_Mangia_Avellaneda_BTC.ipynb # Research notebook with most of what was implemented, with some changes
│   ├── requirements.txt              # Python dependencies
│   └── docker-compose.yml            # Container config for scripts
├── HL_data_collector/                # Separate data collection service
│   ├── hyperliquid_data_collector.py # Market data gathering
│   ├── run_collector.py              # Data collector orchestrator
│   ├── HL_data/                      # Collected market data (CSV files)
│   ├── Dockerfile                    # Data collector container
│   └── requirements.txt              # Python dependencies
├── docker-compose.yml                # Main container orchestration
├── Dockerfile.technical              # Extra python libraries installation in for docker compose
└── show_PnL.py                       # Profit and loss analysis tool
```

## Mathematical Foundation

### Avellaneda-Stoikov Market Making Model

The strategy implements the classical Avellaneda-Stoikov optimal market making model from "High-frequency trading in a limit order book" (2008):

**Core Model Elements:**

| Element | Formula | Interpretation |
|---------|---------|----------------|
| **Mid-price dynamics** | `dS_t = σ dW_t` | Geometric Brownian motion with volatility σ |
| **Order arrivals** | `λ(δ) = A e^(-k δ)` | Exponential intensity function of spread δ |
| **Optimal spreads** | `δ_a = δ_b + (2q + 1)γσ²τ/2` | Asymmetric spreads based on inventory q |
| **Base spread** | `δ* = γσ²τ + (2/γ)ln(1 + γ/k)` | Optimal spread for zero inventory |
| **Inventory** | `q_t`: Accumulated position from filled orders | Running inventory from market making |

**Optimal Bid-Ask Strategy:**
```
Ask Price = S_t + δ_a = S_t + δ*/2 + γσ²τq          
Bid Price = S_t - δ_b = S_t - δ*/2 - γσ²τq         
```

**Key Parameters:**
- `γ`: Risk aversion parameter (inventory penalty coefficient)
- `σ`: Asset price volatility 
- `k`: Order book liquidity parameter (from λ(δ) = A e^(-k δ))
- `τ`: Time remaining until strategy end
- `q`: Current inventory position

### Objective Function and Solution Method

**Market Maker's Optimization Problem:**
```
max E[X_T + Q_T S_T - γ ∫₀ᵀ Q_t² dt]
```
Where:
- `X_T`: Cash accumulated from spread capture
- `Q_T S_T`: Mark-to-market value of final inventory  
- `γ ∫₀ᵀ Q_t² dt`: Inventory holding penalty (risk aversion)

**Solution Method - Dynamic Programming:**
1. **Hamilton-Jacobi-Bellman equation** for value function `H(t,S,q,X)`
2. **Optimal control** determines bid/ask placement as functions of (t,S,q)
3. **Closed-form solution** for optimal spreads under exponential utility

### Parameter Estimation and Calibration

**Gamma (γ) - Risk Aversion:**
- Calculated from optimal portfolio theory and risk preferences
- Controls the trade-off between profit and inventory risk
- Higher γ → tighter spreads around zero inventory

**K Parameter - Order Flow Intensity:**
- Estimated from order book data: `λ(δ) = A e^(-k δ)`
- Fitted using regression on historical order arrival rates vs spread
- Critical for determining optimal spread width

**Sigma (σ) - Volatility:**
- Estimated from recent price returns using rolling windows
- Calculated from high-frequency mid-price movements
- Used for both spread calculation and risk assessment

### Market Regimes and Profitability

**Profitable Conditions:**
- **High order flow** (large A): Many market orders → frequent spread capture
- **Deep order book** (small k): Ability to set wider spreads without affecting fill rates
- **Appropriate volatility**: Sufficient price movement for profitable opportunities
- **Effective risk management**: Proper γ calibration for inventory control

**Strategy Performance Factors:**
- **Spread optimization**: Balance between capture probability and profit per trade
- **Inventory management**: Minimize holding costs while maintaining market presence  
- **Parameter adaptation**: Respond to changing market microstructure
- **Execution quality**: Minimize adverse selection and latency costs

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Hyperliquid API credentials

### Quick Start

1. **Clone and configure:**
   ```bash
   # Configure exchange credentials in user_data/config.json
   # Set your Hyperliquid API keys
   ```

2. **Start data collection:**
   ```bash
   # The HL data collector runs as a separate service
   docker compose up -d hl-collector
   ```
   Will write orderbook, price and orders data flow to files in directory `HL_data_collector/HL_data`

3. **Run the strategy:**
   ```bash
   # cd to root directory of this project  
   docker compose up -d freqtrade_mm
   ```
   Only use in dry-run (paper trading)
   Monitor from Freqtrade web client at http://localhost:3004 (user: MM, pass: MM)

## Configuration

### Main Configuration (`user_data/config.json`)
Currently configured for BTC trading:
```json
{
    "max_open_trades": 1,
    "stake_currency": "USDC", 
    "stake_amount": 20,
    "trading_mode": "futures",
    "exchange": {
        "name": "hyperliquid",
        "pair_whitelist": ["BTC/USDC:USDC"]
    },
    "unfilledtimeout": {
        "entry": 15,
        "exit": 15
    }
}
```

### Parameter Files

The system maintains dynamic parameters in JSON files:

- `scripts/avellaneda_parameters_BTC.json`: Contains γ (gamma), k, and σ (sigma) parameters
- Parameters are calculated by `calculate_avellaneda_parameters.py`
- Updated through `run_avellaneda_param_calculation.py` during strategy execution

## Usage Examples

### Manual Parameter Calculation

```bash
# Calculate Avellaneda parameters for BTC (default)
python scripts/calculate_avellaneda_parameters.py

# For other assets, modify TICKER variable in the script
# Available: BTC, ETH, SOL, WLFI

# View current parameters
cat scripts/avellaneda_parameters_BTC.json
```

## Key Components

### avellaneda.py

The main strategy implementing the Avellaneda-Stoikov model:
- **Dynamic spread calculation** using γ, k, σ parameters
- **Inventory-aware pricing** with asymmetric bid-ask spreads
- **Real-time parameter loading** from `avellaneda_parameters_BTC.json`
- **Automated parameter recalculation** through integrated runner
- **Risk management** with inventory-based spread adjustments

### Parameter Calculation Scripts

- **calculate_avellaneda_parameters.py**: Unified parameter estimation for γ, k, σ
- **run_avellaneda_param_calculation.py**: Strategy-integrated parameter updates
- **Research notebook**: Francesco_Mangia_Avellaneda_BTC.ipynb (parameter analysis and research)

## Risk Management

### Built-in Protections

- **Inventory-based spread adjustment**: Wider spreads for larger positions
- **Position limits**: Single position with $20 USDC stake size
- **Order timeouts**: 15-second unfilled order cancellation  
- **Risk aversion parameter**: γ controls inventory penalty strength
- **Dry-run only**: Currently configured for paper trading safety

## Disclaimer

This software is for educational and research purposes. Market making involves significant financial risk. Always test thoroughly in dry-run mode before deploying with real capital. Past performance does not guarantee future results.
ONLY USE IN DRY-RUN

## License



This project implements academic market making models and is intended for research and educational use.

