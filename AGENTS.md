# Repository Guidelines

## Layout and runtime

- `scripts/` contains the estimator, shared quote policy, public-data replay, evaluator and runnable scientific checks.
- `user_data/config.json` configures one Hyperliquid pair; `user_data/strategies/avellaneda.py` implements the paper-only Freqtrade callbacks. The calculator defaults to that configured pair unless a ticker is supplied.
- Public capture is owned by `../HYPERLIQUID_DATA/hub/`. The parameter service mounts `../HYPERLIQUID_DATA/data/hub/` read-only at `/freqtrade/central-data`; its `HL_DATA_LOC` and shared Python reader are configured in Compose.
- Since 2026-10-09 the local `hl-collector` is stopped behind `legacy-archive`. `runtime/paper/market-data/` and the sibling `data/eth_mm/` are frozen historical sources, including the inputs to older saved sweeps. Do not restart duplicate writers. See [collector and data locations](README.md#collectors-and-data-locations).
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

Grid follow-up (done 2026-11-08): the fixed gamma grid `0.1, 1, 2, 4, 10` × planning horizons `5, 15, 30, 45, 60` minutes, at `1s` and `10s` latency, was repeated on 36 hub days (Oct 2 – Nov 6, holdout from Oct 26), before and after post-only replay, EWMA and the markout term. No cell is profitable. With the markout term the model abstains on PAXG at the 1.5 bp maker fee, and a 98-perp screen found no market with a positive maker edge. Local reports and runnable studies: `runtime/paper/validation/joint_grid_hub_20261107/`, `joint_grid_hub_v2_20261108/` and `market_screen_20261108/` (older: `joint_grid_20261001/`). Rerun the v2 study and the screen only when the fee tier, order size or model inputs change. Keep stake, stops and the 30-minute holding limit fixed, and preserve the 45/60-minute planner-deadline caveat.

Wide-quote follow-up (2026-11-09). Screen v2 (`market_screen_v2_20261108/`) covered 98 perps from the touch to 120 bp: fill value stays at about −4 to −5 bp out to 30 bp, so widening does not help. Only bids 50–120 bp below mid catch wicks that rebound within 15 s, and that region lies beyond the model's 24-h markout support. One candidate is frozen in `wide_bid_useless_20261109/FROZEN.md`: a USELESS bid 80 bp below mid with a best-ask exit. It passed the in-sample kill check narrowly (+11.5 bp/trip at 1 s, +2 bp/trip on the holdout). Its single confirmation runs after 2026-11-26 00:00 UTC on Nov 5–25; follow `RESULTS.md` there and do not change the rule or the pass test.

## Code and delivery

- Use four-space indentation, snake_case functions/variables, PascalCase classes, and uppercase constants/environment variables.
- Reuse the shared loaders and quote policy. Keep units, assumptions and validity limits close to the model; avoid duplicate implementations and parameter-format compatibility code.
- Use `Path` for files and the existing logger for strategy diagnostics. Add a focused runnable check for non-trivial logic.
- Keep commits focused, with short imperative messages and relevant validation evidence. Never commit credentials, generated market data, parameter snapshots, databases or logs.
- GitHub access: the `djienne` SSH key is `~/.ssh/id_rsa_reflechir` (its public key matches GitHub), selected by the `github.com` SSH host configuration. Use `git@github.com:djienne/AVELLANEDA_MARKET_MAKING_FREQTRADE.git` for pushes instead of relying on the active HTTPS login. Never read or publish private-key contents.
- Refer only to the `djienne` GitHub identity in documentation and status messages; do not name other GitHub accounts.
- CPU and API-rate limits are managed by the parent workspace's central tools; do not hand-edit them or touch other bots.
