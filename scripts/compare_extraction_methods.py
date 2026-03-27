import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import BASELINE_RESULTS_DIR, ANALYSIS_RESULTS_DIR


def summarize_file(file_path: Path, method_name: str) -> pd.DataFrame:
    """
    Load one result CSV and return a summary table with:
    - overall match rate
    - per-variant match rate
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing results file: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Results file is empty: {file_path}")

    if "exact_match_normalized" not in df.columns:
        raise ValueError(
            f"'exact_match_normalized' column not found in: {file_path}"
        )

    summary_rows = []

    # Overall summary
    summary_rows.append(
        {
            "method": method_name,
            "scope": "overall",
            "variant": "all",
            "total_cases": len(df),
            "matches": int(df["exact_match_normalized"].sum()),
            "match_rate": float(df["exact_match_normalized"].mean()),
        }
    )

    # Per-variant summary
    variant_summary = (
        df.groupby("variant")
        .agg(
            total_cases=("variant", "count"),
            matches=("exact_match_normalized", "sum"),
            match_rate=("exact_match_normalized", "mean"),
        )
        .reset_index()
    )

    for _, row in variant_summary.iterrows():
        summary_rows.append(
            {
                "method": method_name,
                "scope": "variant",
                "variant": row["variant"],
                "total_cases": int(row["total_cases"]),
                "matches": int(row["matches"]),
                "match_rate": float(row["match_rate"]),
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    gemini_path = BASELINE_RESULTS_DIR / "baseline_results.csv"
    raw_ocr_path = ANALYSIS_RESULTS_DIR / "ocr_test_results.csv"
    normalized_ocr_path = ANALYSIS_RESULTS_DIR / "ocr_oracle_normalized_results.csv"

    gemini_summary = summarize_file(gemini_path, "gemini_baseline")
    raw_ocr_summary = summarize_file(raw_ocr_path, "raw_ocr")
    normalized_ocr_summary = summarize_file(
        normalized_ocr_path, "ocr_oracle_normalized"
    )

    combined_summary = pd.concat(
        [gemini_summary, raw_ocr_summary, normalized_ocr_summary],
        ignore_index=True,
    )

    output_path = ANALYSIS_RESULTS_DIR / "extraction_method_comparison.csv"
    combined_summary.to_csv(output_path, index=False)

    print("=== Combined Summary ===")
    print(combined_summary)
    print()

    print("=== Overall Comparison ===")
    overall_df = combined_summary[combined_summary["scope"] == "overall"].copy()
    print(overall_df[["method", "total_cases", "matches", "match_rate"]])
    print()

    print("=== Per-Variant Comparison ===")
    variant_df = combined_summary[combined_summary["scope"] == "variant"].copy()
    print(variant_df[["method", "variant", "total_cases", "matches", "match_rate"]])
    print()

    print(f"Saved comparison summary to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)