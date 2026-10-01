"""Inspect the reference midpoint used by the estimator and strategy."""
import argparse
from utils import load_book_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orderbooks")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    book = load_book_data(args.orderbooks, args.start, args.end, depth=1)
    print(book[["bid_price_0", "ask_price_0", "mid_price"]].describe())
