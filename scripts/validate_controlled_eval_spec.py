import sys
from pathlib import Path

import pandas as pd

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROMPTS_DIR


REQUIRED_COLUMNS = [
    "case_id",
    "intent_family",
    "redacted_eval_prompt",
    "proxy_safe_prompt",
    "expected_guardrail_decision",
    "enabled",
    "notes",
]


def main():
    csv_path = PROMPTS_DIR / "controlled_eval_spec.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Controlled eval spec not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=== Loaded Controlled Evaluation Spec ===")
    print(csv_path)
    print()

    print("=== First 5 Rows ===")
    print(df.head())
    print()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    duplicate_ids = df[df["case_id"].duplicated()]["case_id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate case IDs found: {duplicate_ids}")

    total_rows = len(df)
    enabled_rows = int((df["enabled"] == True).sum())

    print("=== Summary ===")
    print(f"Total controlled cases: {total_rows}")
    print(f"Enabled rows: {enabled_rows}")
    print(f"Intent families: {df['intent_family'].nunique()}")
    print()

    print("=== Case Counts by Expected Guardrail Decision ===")
    print(df["expected_guardrail_decision"].value_counts(dropna=False))
    print()

    print("=== Intent Families ===")
    print(sorted(df["intent_family"].unique().tolist()))
    print()

    print("Validation successful.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)