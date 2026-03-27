import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import ANALYSIS_RESULTS_DIR


def main():
    pipeline_path = ANALYSIS_RESULTS_DIR / "guardrail_pipeline_test_results.csv"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline results file not found: {pipeline_path}")

    df = pd.read_csv(pipeline_path)

    if df.empty:
        raise ValueError("guardrail_pipeline_test_results.csv is empty")

    print("=== Loaded Guardrail Pipeline Results ===")
    print(pipeline_path)
    print()

    print("=== First 5 Rows ===")
    print(df.head())
    print()

    total_cases = len(df)
    allow_count = int((df["decision"] == "allow").sum())
    review_count = int((df["decision"] == "review").sum())
    block_count = int((df["decision"] == "block").sum())

    allow_rate = allow_count / total_cases
    review_rate = review_count / total_cases
    block_rate = block_count / total_cases

    print("=== Overall Decision Summary ===")
    print(f"Total benign cases: {total_cases}")
    print(f"Allow:  {allow_count} ({allow_rate:.2%})")
    print(f"Review: {review_count} ({review_rate:.2%})")
    print(f"Block:  {block_count} ({block_rate:.2%})")
    print()

    # Treat review or block on benign inputs as false-positive-style flags
    flagged_benign_df = df[df["decision"].isin(["review", "block"])].copy()

    flagged_rate = len(flagged_benign_df) / total_cases

    print("=== Benign Flagging Summary ===")
    print(f"Flagged benign cases (review or block): {len(flagged_benign_df)}")
    print(f"Flagged benign rate: {flagged_rate:.2%}")
    print()

    # By variant summary
    variant_summary = (
        df.groupby(["true_variant", "decision"])
        .size()
        .reset_index(name="count")
    )

    print("=== Decision Counts by Variant ===")
    print(variant_summary)
    print()

    variant_summary_path = ANALYSIS_RESULTS_DIR / "guardrail_pipeline_analysis_variant_summary.csv"
    variant_summary.to_csv(variant_summary_path, index=False)

    # Detailed flagged benign rows
    print("=== Flagged Benign Cases ===")
    if flagged_benign_df.empty:
        print("No flagged benign cases found.")
    else:
        print(
            flagged_benign_df[
                [
                    "id",
                    "true_variant",
                    "expected_text",
                    "extracted_text",
                    "matched_keywords",
                    "risk_score",
                    "risk_level",
                    "decision",
                ]
            ].to_string(index=False)
        )
    print()

    flagged_benign_path = ANALYSIS_RESULTS_DIR / "guardrail_flagged_benign_cases.csv"
    flagged_benign_df.to_csv(flagged_benign_path, index=False)

    # Prompt-level analysis: do the same prompt IDs get flagged across all variants?
    prompt_flag_summary = (
        flagged_benign_df.groupby("id")
        .size()
        .reset_index(name="num_flagged_variants")
    )

    prompt_flag_summary_path = ANALYSIS_RESULTS_DIR / "guardrail_flagged_prompt_summary.csv"
    prompt_flag_summary.to_csv(prompt_flag_summary_path, index=False)

    print("=== Saved Analysis Files ===")
    print(f"Variant summary: {variant_summary_path}")
    print(f"Flagged benign cases: {flagged_benign_path}")
    print(f"Flagged prompt summary: {prompt_flag_summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)