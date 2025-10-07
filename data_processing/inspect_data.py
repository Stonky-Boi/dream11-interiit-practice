import pandas as pd
import argparse
from pathlib import Path
import sys

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

def inspect_csv_file(file_path, export_excel=False):
    """
    Loads a CSV file, prints a summary, and optionally exports it to Excel.
    """
    if not file_path.exists():
        print(f"❌ Error: File not found at '{file_path}'")
        sys.exit(1)

    print(f"\n--- 🕵️♂️ Inspecting: {file_path} ---\n")

    df = pd.read_csv(file_path)  # CHANGED to read_csv

    print(f"## 1. Shape: Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    print(f"## 2. Columns and Data Types:")
    df.info()
    print("\n")

    print(f"## 3. Random Sample of 20 Rows:")
    print(df.sample(n=min(20, len(df))))
    print("\n" + "-"*50 + "\n")

    if export_excel:
        tmp_dir = Path('data/tmp')
        tmp_dir.mkdir(exist_ok=True)
        excel_path = tmp_dir / f"{file_path.stem}.xlsx"
        print(f"## 4. Exporting to Excel...")
        df.to_excel(excel_path, index=False)
        print(f"✅ Exported to: {excel_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a CSV file and optionally export to Excel.")
    parser.add_argument('file_path', type=str, help="Path to the CSV file")
    parser.add_argument('--export-excel', action='store_true', help="Export to Excel")
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    inspect_csv_file(file_path, export_excel=args.export_excel)
