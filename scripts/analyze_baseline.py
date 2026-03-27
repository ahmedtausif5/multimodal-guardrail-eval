import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import BASELINE_RESULTS_DIR, ANALYSIS_RESULTS_DIR


def main():
    baseline_path = BASELINE_RESULTS_DIR / "baseline_results.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results file not found: {baseline_path}")

    df = pd.read_csv(baseline_path)

    if df.empty:
        raise ValueError("baseline_results.csv is empty")

    print("=== Loaded Baseline Results ===")
    print(baseline_path)
    print()

    print("=== First 5 Rows ===")
    print(df.head())
    print()

    # Overall summary
    total_cases = len(df)
    total_matches = int(df["exact_match_normalized"].sum())
    total_mismatches = total_cases - total_matches
    overall_match_rate = df["exact_match_normalized"].mean()

    print("=== Overall Summary ===")
    print(f"Total cases: {total_cases}")
    print(f"Matches: {total_matches}")
    print(f"Mismatches: {total_mismatches}")
    print(f"Overall normalized exact match rate: {overall_match_rate:.2%}")
    print()

    # Accuracy by variant
    variant_summary = (
        df.groupby("variant")
        .agg(
            total_cases=("id", "count"),
            matches=("exact_match_normalized", "sum"),
            match_rate=("exact_match_normalized", "mean"),
        )
        .reset_index()
    )

    print("=== Match Rate by Variant ===")
    print(variant_summary)
    print()

    # Save variant summary
    variant_summary_path = ANALYSIS_RESULTS_DIR / "baseline_variant_summary.csv"
    variant_summary.to_csv(variant_summary_path, index=False)

    # Isolate mismatches
    mismatches_df = df[df["exact_match_normalized"] == False].copy()

    print("=== Mismatched Cases ===")
    if mismatches_df.empty:
        print("No mismatches found.")
    else:
        print(
            mismatches_df[
                ["id", "variant", "expected_text", "model_output"]
            ].to_string(index=False)
        )
    print()

    mismatches_path = ANALYSIS_RESULTS_DIR / "baseline_mismatches.csv"
    mismatches_df.to_csv(mismatches_path, index=False)

    print("=== Saved Analysis Files ===")
    print(f"Variant summary: {variant_summary_path}")
    print(f"Mismatches: {mismatches_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)