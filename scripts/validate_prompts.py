import sys
from pathlib import Path

import pandas as pd

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROMPTS_DIR


REQUIRED_COLUMNS = [
    "id",
    "prompt_text",
    "category",
    "is_benign",
    "enabled",
    "notes",
]


def main():
    csv_path = PROMPTS_DIR / "curated_prompts.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=== Loaded Prompt File ===")
    print(csv_path)
    print()

    print("=== First 5 Rows ===")
    print(df.head())
    print()

    # Check required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check duplicate IDs
    duplicate_ids = df[df["id"].duplicated()]["id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate IDs found: {duplicate_ids}")

    # Basic counts
    total_rows = len(df)
    enabled_rows = df[df["enabled"] == True]
    benign_rows = df[df["is_benign"] == True]
    risky_rows = df[df["is_benign"] == False]

    print("=== Summary ===")
    print(f"Total rows: {total_rows}")
    print(f"Enabled rows: {len(enabled_rows)}")
    print(f"Benign rows: {len(benign_rows)}")
    print(f"Non-benign rows: {len(risky_rows)}")
    print()

    print("=== Enabled IDs ===")
    print(enabled_rows["id"].tolist())
    print()

    print("Validation successful.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)