import os
from hyperliquid_data_collector import HyperliquidDataCollector

def _symbols():
    raw = os.getenv("SYMBOLS", "BTC,ETH,SOL,WLFI")
    return [s.strip() for s in raw.split(",") if s.strip()]

def main():
    symbols = _symbols()
    output_dir = os.getenv("OUTPUT_DIR", "HL_data")
    try:
        orderbook_depth = int(os.getenv("ORDERBOOK_DEPTH", "20"))
    except ValueError:
        orderbook_depth = 20

    print("Hyperliquid Tick Data Collector (Docker)")
    print("========================================")
    print(f"Symbols:         {symbols}")
    print(f"Output dir:      {output_dir}")
    print(f"Orderbook depth: {orderbook_depth}")

    collector = HyperliquidDataCollector(
        symbols, output_dir, orderbook_depth=orderbook_depth
    )
    collector.start_collection()

if __name__ == "__main__":
    main()

