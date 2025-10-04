import pandas as pd
import argparse
from pathlib import Path
import sys

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def inspect_parquet_file(file_path, export_csv=False):
    """
    Loads a Parquet file, prints a summary, and optionally exports it to CSV.
    """
    if not file_path.exists():
        print(f"❌ Error: File not found at '{file_path}'")
        sys.exit(1)
    print(f"\n--- 🕵️‍♂️ Inspecting: {file_path} ---\n")
    df = pd.read_parquet(file_path)

    # --- Print Summary Info ---
    print(f"## 1. Shape: Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    print(f"## 2. Columns and Data Types:")
    df.info()
    print("\n")

    # --- Print a Larger Random Sample ---
    print(f"## 3. Random Sample of 20 Rows:")
    print(df.sample(n=min(20, len(df))))
    print("\n" + "-"*50 + "\n")

    # --- Optional CSV Export ---
    if export_csv:
        tmp_dir = Path('data/tmp')
        tmp_dir.mkdir(exist_ok=True)
        csv_path = tmp_dir / f"{file_path.stem}.csv"    
        print(f"## 4. Exporting to CSV for easy viewing...")
        df.to_csv(csv_path, index=False)
        print(f"✅ Success! File exported to: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect Parquet data files.")
    parser.add_argument('--file', type=str, required=True, choices=['interim', 'processed'], help="Specify which file to inspect: 'interim' or 'processed'.")
    parser.add_argument('--export_to_csv', action='store_true', help="If set, exports the Parquet file to a CSV in the data/tmp/ directory.")
    args = parser.parse_args()

    paths = {
        'interim': Path('data/interim/player_match_stats.parquet'),
        'processed': Path('data/processed/final_model_data.parquet')
    }
    inspect_parquet_file(paths[args.file], args.export_to_csv)