import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROMPTS_DIR


def main():
    spec_path = PROMPTS_DIR / "controlled_eval_spec.csv"

    if not spec_path.exists():
        raise FileNotFoundError(f"Controlled eval spec not found: {spec_path}")

    df = pd.read_csv(spec_path)

    if df.empty:
        raise ValueError("controlled_eval_spec.csv is empty")

    required_columns = [
        "case_id",
        "intent_family",
        "proxy_safe_prompt",
        "expected_guardrail_decision",
        "notes",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in controlled_eval_spec.csv: {missing}")

    proxy_df = pd.DataFrame(
        {
            "id": df["case_id"],
            "prompt_text": df["proxy_safe_prompt"],
            "category": df["intent_family"],
            "is_benign": False,
            "enabled": True,
            "expected_guardrail_decision": df["expected_guardrail_decision"],
            "notes": df["notes"],
        }
    )

    output_path = PROMPTS_DIR / "controlled_proxy_prompts.csv"
    proxy_df.to_csv(output_path, index=False)

    print("=== Built Controlled Proxy Dataset ===")
    print(f"Input spec: {spec_path}")
    print(f"Output dataset: {output_path}")
    print()

    print("=== First 5 Rows ===")
    print(proxy_df.head())
    print()

    print("=== Summary ===")
    print(f"Total rows: {len(proxy_df)}")
    print(f"Enabled rows: {(proxy_df['enabled'] == True).sum()}")
    print(f"Unique categories: {proxy_df['category'].nunique()}")
    print()

    print("=== Expected Guardrail Decision Counts ===")
    print(proxy_df["expected_guardrail_decision"].value_counts(dropna=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)