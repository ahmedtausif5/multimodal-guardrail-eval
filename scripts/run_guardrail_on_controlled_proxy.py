import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROMPTS_DIR, ANALYSIS_RESULTS_DIR
from guardrail_extractor import extract_text_with_candidate_normalization
from guardrail_screener import score_text_risk


def main():
    metadata_path = PROMPTS_DIR / "controlled_proxy_rendered_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Controlled proxy rendered metadata file not found: {metadata_path}"
        )

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("controlled_proxy_rendered_metadata.csv is empty")

    results = []

    print("=== Running Guardrail Pipeline on Controlled Proxy Benchmark ===")
    print("Pipeline: image -> extraction -> screening -> decision")
    print(f"Loaded metadata from: {metadata_path}")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        variant = row["variant"]
        image_path = row["image_path"]
        prompt_text = row["prompt_text"]
        category = row["category"]
        expected_guardrail_decision = row["expected_guardrail_decision"]

        print(f"[{i + 1}/{len(df)}] Processing {prompt_id} | {variant} | {category}")

        extraction = extract_text_with_candidate_normalization(image_path)
        screening = score_text_risk(extraction["extracted_text"])

        result_row = {
            "id": prompt_id,
            "variant": variant,
            "category": category,
            "image_path": image_path,
            "prompt_text": prompt_text,
            "expected_guardrail_decision": expected_guardrail_decision,
            "best_transform_chosen": extraction["best_transform_chosen"],
            "extracted_text": extraction["extracted_text"],
            "extracted_text_normalized": extraction["extracted_text_normalized"],
            "best_avg_confidence": extraction["best_avg_confidence"],
            "matched_keywords": ", ".join(screening["matched_keywords"]),
            "risk_score": screening["risk_score"],
            "risk_level": screening["risk_level"],
            "actual_guardrail_decision": screening["decision"],
            "decision_matches_expectation": screening["decision"] == expected_guardrail_decision,
        }

        results.append(result_row)

    results_df = pd.DataFrame(results)

    output_path = ANALYSIS_RESULTS_DIR / "controlled_proxy_guardrail_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved results to: {output_path}")
    print()

    print("=== Actual Guardrail Decision Counts ===")
    print(results_df["actual_guardrail_decision"].value_counts(dropna=False))
    print()

    print("=== Decision Match vs Expectation ===")
    print(results_df["decision_matches_expectation"].value_counts(dropna=False))
    print()

    summary_by_variant = (
        results_df.groupby(["variant", "actual_guardrail_decision"])
        .size()
        .reset_index(name="count")
    )

    summary_by_category = (
        results_df.groupby(["category", "actual_guardrail_decision"])
        .size()
        .reset_index(name="count")
    )

    variant_summary_path = ANALYSIS_RESULTS_DIR / "controlled_proxy_guardrail_variant_summary.csv"
    category_summary_path = ANALYSIS_RESULTS_DIR / "controlled_proxy_guardrail_category_summary.csv"

    summary_by_variant.to_csv(variant_summary_path, index=False)
    summary_by_category.to_csv(category_summary_path, index=False)

    print("=== Saved Analysis Files ===")
    print(f"Detailed results: {output_path}")
    print(f"Variant summary: {variant_summary_path}")
    print(f"Category summary: {category_summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)