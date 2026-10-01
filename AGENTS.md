# Repository Guidelines

## Layout and runtime

- `scripts/` contains the estimator, shared quote policy, public-data replay, evaluator and runnable scientific checks.
- `user_data/config.json` configures one Hyperliquid pair; `user_data/strategies/avellaneda.py` implements the paper-only Freqtrade callbacks. The calculator defaults to that configured pair unless a ticker is supplied.
- `HL_data_collector/run_collector.py` captures public streams. The collector, calculator and bot share the image built by `Dockerfile.technical`.
- The active paper collector writes `runtime/paper/market-data/`; this is the input used by the parameter service and saved sweeps. A separate shared collector writes `../HYPERLIQUID_DATA/data/eth_mm/`, with source in `../Cartea-Jaimungal_MARKET_MAKING_FREQTRADE/scripts/`. See [collector and data locations](README.md#collectors-and-data-locations) before choosing a backtest archive.
- `docker-compose.yml` defines the three services. The isolated experiment uses `compose.paper.yml` and project name `avellaneda-paper`; follow the build/start commands in [README.md](README.md#start).
- Paper files live under gitignored `runtime/paper/`: `market-data/`, `params/` and `state/`. The latter holds `trades.sqlite`, `trial.json` and `freqtrade.log`. Default Compose paths use `runtime/main/`.
- Credentials come from gitignored `.env`. The calculator publishes every fifteen minutes when valid and retries failed calculations after one minute.

## Validation

Use Docker Compose for checks; do not rely on a different host Python environment.

```bash
docker compose -p avellaneda-paper -f docker-compose.yml -f compose.paper.yml run --rm --no-deps --entrypoint python hl-params /freqtrade/scripts/check_pipeline.py
```

After estimator changes, also run it on captured data, keeping diagnostic output separate from the running writer:

```bash
docker compose -p avellaneda-paper -f docker-compose.yml -f compose.paper.yml run --rm --no-deps --entrypoint python hl-params /freqtrade/scripts/calculate_avellaneda_parameters.py ETH --output-dir /freqtrade/params/diagnostics
```

Inspect the result and blocking reasons. Insufficient data is inconclusive; tests and a running process do not establish profitability. For runtime changes, verify collector health, parameters, trial state and Compose logs. Preserve the paper database and risk-stop state across restarts.

Future research follow-up: when materially more usable data has accumulated, repeat the fixed gamma grid `0.1, 1, 2, 4, 10` crossed with planning horizons `5, 15, 30, 45, 60` minutes, at both `1s` and `10s` latency. Keep stake, stops and the actual 30-minute holding limit fixed. Compare PnL, drawdown, completed trades, holding times and maker/taker fractions on new chronological data. Preserve the 45/60-minute planner-deadline caveat. The local baseline, full results and runnable study are in `runtime/paper/validation/joint_grid_20261001/`; the old 2.5-hour sample is not a fresh holdout.

## Code and delivery

- Use four-space indentation, snake_case functions/variables, PascalCase classes, and uppercase constants/environment variables.
- Reuse the shared loaders and quote policy. Keep units, assumptions and validity limits close to the model; avoid duplicate implementations and parameter-format compatibility code.
- Use `Path` for files and the existing logger for strategy diagnostics. Add a focused runnable check for non-trivial logic.
- Keep commits focused, with short imperative messages and relevant validation evidence. Never commit credentials, generated market data, parameter snapshots, databases or logs.
- GitHub access: the `djienne` SSH key is `~/.ssh/id_rsa_reflechir` (its public key matches GitHub), selected by the `github.com` SSH host configuration. Use `git@github.com:djienne/AVELLANEDA_MARKET_MAKING_FREQTRADE.git` for pushes instead of relying on the active HTTPS login. Never read or publish private-key contents.
- Refer only to the `djienne` GitHub identity in documentation and status messages; do not name other GitHub accounts.
- CPU and API-rate limits are managed by the parent workspace's central tools; do not hand-edit them or touch other bots.
