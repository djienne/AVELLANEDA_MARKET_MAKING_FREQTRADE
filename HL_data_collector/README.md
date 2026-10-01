# Hyperliquid public-data collector

This collector uses the project's shared Docker image and dependencies. Run it from the repository root:

```bash
docker compose -p avellaneda-paper -f docker-compose.yml -f compose.paper.yml up -d hl-collector
docker compose -p avellaneda-paper -f docker-compose.yml -f compose.paper.yml logs -f hl-collector
```

The default subscriptions are PAXG and ETH. The fast L2 stream supplies five levels; BBO, trades and asset contexts are separate subscriptions. The normal twenty-level stream was measured at roughly five-second updates, which is too slow for this experiment's two-second freshness bound.

| Environment variable | Default |
| :--- | :--- |
| `SYMBOLS` | `PAXG,ETH` |
| `OUTPUT_DIR` | `HL_data` outside Compose; `/freqtrade/market-data` in Compose |
| `ORDERBOOK_DEPTH` | `5` |

`run_collector.py` is the Python entry point. Dependencies are pinned in the root project's `scripts/requirements.txt`.

## Storage and health

Each stream has a `{kind}_{SYMBOL}.parquet/` directory. Kinds are `orderbooks`, `trades`, `prices`, `contexts` and `funding`.

- Exchange time and local receipt time are preserved.
- Schemas use explicit nullable types, including temporarily absent depth levels.
- A batch is published after five minutes or 10,000 rows. Only closed, atomically renamed parquet files are visible to readers.
- Failed writes retain their pending records. Buffer overflow is fatal and reported, rather than silently discarding old observations.
- Settled funding history is separate from the current funding prediction in asset contexts.
- Timestamped market metadata records quantity precision, minimum size and public maker/taker fees.
- `health.json` reports subscriptions, per-symbol book freshness, pending/published counts and errors.
- Disconnects and stalled feeds trigger resubscription, with ten-second connection timeouts and two-to-thirty-second retry backoff. Old freshness is cleared on reconnect.
- Graceful shutdown flushes buffered records. After an abrupt crash, published batches remain readable; the unpublished batch can be lost, and temporary files are ignored.

The calculator also validates event timestamps, coverage and trade identities. An empty trade window is only usable when its surrounding capture is valid. Public streams do not expose an authenticated order's queue position.

The paper stack writes to `runtime/paper/market-data/`. Existing collectors and historical datasets are separate and are not modified.
