import pandas as pd
import argparse
from pathlib import Path
import sys

def inspect_parquet_file(file_path):
    """
    Loads a Parquet file and prints a summary of its contents.
    
    Args:
        file_path (Path): The path to the Parquet file.
    """
    if not file_path.exists():
        print(f"❌ Error: File not found at '{file_path}'")
        print("Please ensure you have run the data processing and feature engineering scripts first.")
        sys.exit(1)

    print(f"\n--- 🕵️‍♂️ Inspecting: {file_path} ---\n")
    
    df = pd.read_parquet(file_path)

    # --- Print Shape ---
    print(f"1. Shape")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # --- Print Data Types & Non-Null Counts ---
    print(f"2. Columns and Data Types")
    df.info()
    print("\n")

    # --- Print Head ---
    print(f"3. First 5 Rows (Head)")
    print(df.head())
    print("\n")

    # --- Print Tail ---
    print(f"4. Last 5 Rows (Tail)")
    print("Useful for checking the most recent data and engineered features.")
    print(df.tail())
    print("\n" + "-"*50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect interim or processed Parquet data files.")
    parser.add_argument('--file', type=str, required=True, choices=['interim', 'processed'],
                        help="Specify which file to inspect: 'interim' or 'processed'.")
    
    args = parser.parse_args()

    # Define file paths based on project structure
    paths = {
        'interim': Path('data/interim/player_match_stats.parquet'),
        'processed': Path('data/processed/final_model_data.parquet')
    }

    inspect_parquet_file(paths[args.file])