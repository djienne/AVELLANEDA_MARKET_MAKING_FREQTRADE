# Repository Guidelines

## Project Structure & Module Organization
- `scripts/` contains parameter calculators/backtesting (`calculate_avellaneda_parameters.py`, `volatility.py`, `intensity.py`) writing `avellaneda_parameters_{TICKER}.json` to `scripts/` unless `AVELLANEDA_PARAMS_DIR` is set.
- `user_data/` hosts Freqtrade config (`config.json`, `config_short.json`), strategy logic in `strategies/avellaneda.py`, and runtime artifacts (`logs/`, `tradesv3.sqlite`, `data/`, `hyperopt_results/`).
- `HL_data_collector/` captures Hyperliquid streams via `run_collector.py`, persisting parquet files under `HL_data_collector/HL_data/`.
- Root utilities: `docker-compose.yml` wires `freqtrade_mm` + `hl-collector`; `Dockerfile.technical` extends the bot image; `show_PnL.py` and `test_env.py` are local diagnostics.

## Build, Test, and Development Commands
- `docker-compose build` builds the freqtrade bot image (with `Dockerfile.technical`) and the collector.
- `docker-compose up` starts the bot (using `user_data/config.json` + `strategies/avellaneda.py`) and the data collector with persistent host volumes; use `docker-compose down` to stop/clean containers.
- `python scripts/calculate_avellaneda_parameters.py PAXG --minutes 15` recomputes Avellaneda parameters from `HL_data_collector/HL_data` and emits `scripts/avellaneda_parameters_PAXG.json`; override output with `AVELLANEDA_PARAMS_DIR`.
- `python HL_data_collector/run_collector.py` runs the collector outside Docker; configure with `SYMBOLS`, `OUTPUT_DIR`, and `ORDERBOOK_DEPTH` env vars.
- `python test_env.py` quickly verifies key numeric dependencies; `python show_PnL.py` inspects stored trades.

## Coding Style & Naming Conventions
- Python 3.x with 4-space indentation; prefer type hints and `pathlib.Path` for file handling.
- Use snake_case for functions/variables, PascalCase for classes, and uppercase for constants/env keys.
- Keep strategy parameters in JSON named `avellaneda_parameters_{TICKER}.json` (uppercase ticker) to align with config whitelists.
- Log with the existing `logging` setup; avoid ad-hoc prints in strategy code.

## Testing Guidelines
- Minimal automated coverage exists; for quick checks run `python user_data/strategies/test_load_ave_config.py` to validate parameter file discovery.
- When modifying parameter generation, run `python scripts/calculate_avellaneda_parameters.py ETH` and confirm the summary plus JSON output looks sane.
- For end-to-end validation, start `docker-compose up` in dry-run mode and watch `user_data/logs/` for clean startup (no stack traces).

## Commit & Pull Request Guidelines
- Follow the repo's short, imperative commit style (`Update README.md`, `Add ETH parameters`); keep scope focused and messages under ~72 chars.
- In PRs, describe the intent, list commands/logs run (e.g., `docker-compose up`, parameter calculator output), and link any related issue or trading-pair change.
- Include screenshots or log snippets when altering strategy behavior, configs, or collector settings; avoid committing generated data (`HL_data_collector/HL_data`, `tradesv3.sqlite*`, large logs).

## Security & Configuration Tips
- Do not commit secrets or API keys; rely on env vars (`HL_DATA_LOC`, `AVELLANEDA_PARAMS_DIR`, `SYMBOLS`, `OUTPUT_DIR`) and keep them out of git.
- Generated parquet data and SQLite trade logs may contain sensitive trading history; treat them as local-only and gitignored by default.
