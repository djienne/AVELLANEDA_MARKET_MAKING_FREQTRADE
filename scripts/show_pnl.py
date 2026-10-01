"""Print Freqtrade's native profit summary; run inside the bot container for API credentials.

These are framework PnL metrics. The liquidation-equity risk stop is recorded in trial.json.
"""
import json
import os

from freqtrade_client import FtRestClient


if __name__ == "__main__":
    client = FtRestClient("http://127.0.0.1:8080", os.environ["FREQTRADE__API_SERVER__USERNAME"],
                          os.environ["FREQTRADE__API_SERVER__PASSWORD"])
    profit = client.profit()
    if not isinstance(profit, dict) or "profit_all_coin" not in profit:
        raise SystemExit("Profit summary unavailable; check the bot API and its credentials")
    print(json.dumps(profit, indent=2))
