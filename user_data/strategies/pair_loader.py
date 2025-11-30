"""Pair and ticker helpers for strategy and tooling.

Reads the first pair in `user_data/config.json` (or `config.json` when run from the
project root) so changing `exchange.pair_whitelist` propagates to strategy and
parameter generation without further edits.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PAIR = "PAXG/USDC:USDC"
CONFIG_FILENAMES = ["user_data/config.json", "config.json"]


def _find_upwards(filename: str, start: Path, max_up: int = 6) -> Optional[Path]:
    """Search upward from `start` for `filename`; return path or None."""
    current = start.resolve()
    for _ in range(max_up + 1):
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def get_config_path(start_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate the primary config file by walking upward from the given start."""
    if start_dir is None:
        try:
            start_dir = Path(__file__).resolve().parent
        except Exception:
            start_dir = Path.cwd()

    for name in CONFIG_FILENAMES:
        found = _find_upwards(name, start_dir)
        if found:
            return found
    return None


def load_config(path: Optional[Path] = None) -> dict:
    """Load config JSON; return empty dict if missing or invalid."""
    if path is None:
        path = get_config_path()
    if path is None or not path.exists():
        logger.warning("Config file not found; falling back to defaults.")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # JSON errors or IO errors
        logger.error("Failed to read config %s: %s", path, exc)
        return {}


def get_active_pair(default_pair: str = DEFAULT_PAIR) -> str:
    """Return the first whitelisted pair, or the provided default."""
    cfg = load_config()
    pair_list = cfg.get("exchange", {}).get("pair_whitelist") or []
    if pair_list and isinstance(pair_list, list):
        return str(pair_list[0])
    return default_pair


def pair_to_ticker(pair: str) -> str:
    """Extract base ticker from pair strings like 'PAXG/USDC:USDC'."""
    if not pair:
        return ""
    base = pair.split("/")[0]
    base = base.split(":")[0]
    return base.upper()


def get_param_filename(pair: Optional[str] = None) -> str:
    """Return the expected parameter filename for a given pair/ticker."""
    ticker = pair_to_ticker(pair or get_active_pair()) or pair_to_ticker(DEFAULT_PAIR)
    return f"avellaneda_parameters_{ticker}.json"


__all__ = [
    "get_active_pair",
    "pair_to_ticker",
    "get_param_filename",
    "get_config_path",
    "load_config",
]
