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
    "expected_guardrail_decision",
    "notes",
]


def main():
    csv_path = PROMPTS_DIR / "controlled_proxy_prompts.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Controlled proxy prompt file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=== Loaded Controlled Proxy Prompt Dataset ===")
    print(csv_path)
    print()

    print("=== First 5 Rows ===")
    print(df.head())
    print()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    duplicate_ids = df[df["id"].duplicated()]["id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate IDs found: {duplicate_ids}")

    print("=== Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Enabled rows: {(df['enabled'] == True).sum()}")
    print(f"Unique categories: {df['category'].nunique()}")
    print()

    print("=== Expected Guardrail Decision Counts ===")
    print(df["expected_guardrail_decision"].value_counts(dropna=False))
    print()

    print("Validation successful.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)