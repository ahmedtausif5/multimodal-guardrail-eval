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
    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Rendered metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("rendered_images_metadata.csv is empty")

    results = []

    print("=== Testing Full Guardrail Pipeline ===")
    print("Pipeline: image -> extraction -> screening -> decision")
    print(f"Loaded metadata from: {metadata_path}")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        true_variant = row["variant"]
        image_path = row["image_path"]
        expected_text = row["prompt_text"]

        print(f"[{i + 1}/{len(df)}] Processing {prompt_id} | {true_variant}")

        extraction = extract_text_with_candidate_normalization(image_path)
        screening = score_text_risk(extraction["extracted_text"])

        result_row = {
            "id": prompt_id,
            "true_variant": true_variant,
            "image_path": image_path,
            "expected_text": expected_text,
            "best_transform_chosen": extraction["best_transform_chosen"],
            "extracted_text": extraction["extracted_text"],
            "extracted_text_normalized": extraction["extracted_text_normalized"],
            "best_avg_confidence": extraction["best_avg_confidence"],
            "matched_keywords": ", ".join(screening["matched_keywords"]),
            "risk_score": screening["risk_score"],
            "risk_level": screening["risk_level"],
            "decision": screening["decision"],
        }

        results.append(result_row)

    results_df = pd.DataFrame(results)

    output_path = ANALYSIS_RESULTS_DIR / "guardrail_pipeline_test_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved pipeline results to: {output_path}")
    print()

    print("=== Decision Counts ===")
    decision_counts = results_df["decision"].value_counts(dropna=False)
    print(decision_counts)
    print()

    print("=== Decision Counts by Variant ===")
    decision_variant_summary = (
        results_df.groupby(["true_variant", "decision"])
        .size()
        .reset_index(name="count")
    )
    print(decision_variant_summary)
    print()

    summary_path = ANALYSIS_RESULTS_DIR / "guardrail_pipeline_decision_summary.csv"
    decision_variant_summary.to_csv(summary_path, index=False)

    print("=== Saved Analysis Files ===")
    print(f"Detailed pipeline results: {output_path}")
    print(f"Decision summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)