import os
import pyarrow.parquet as pq
import sys

def check_parquet_file(filepath):
    print(f"Checking {filepath}...")
    try:
        pq.read_table(filepath)
        print(f"  OK: {filepath}")
        return True
    except Exception as e:
        print(f"  ERROR: {filepath} - {e}")
        return False

def main():
    base_dir = "HL_data_collector/HL_data"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".parquet"):
                filepath = os.path.join(root, file)
                check_parquet_file(filepath)

if __name__ == "__main__":
    main()
