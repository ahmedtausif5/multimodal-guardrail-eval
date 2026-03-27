import sys
from pathlib import Path

import pandas as pd

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROMPTS_DIR, ANALYSIS_RESULTS_DIR
from guardrail_extractor import (
    extract_text_with_candidate_normalization,
    normalize_text,
)


def main():
    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Rendered metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("rendered_images_metadata.csv is empty")

    results = []

    print("=== Testing Reusable Guardrail Extractor ===")
    print(f"Loaded metadata from: {metadata_path}")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        true_variant = row["variant"]
        image_path = row["image_path"]
        expected_text = row["prompt_text"]

        print(f"[{i + 1}/{len(df)}] Extracting {prompt_id} | {true_variant}")

        extraction = extract_text_with_candidate_normalization(image_path)

        result_row = {
            "id": prompt_id,
            "true_variant": true_variant,
            "image_path": image_path,
            "expected_text": expected_text,
            "best_transform_chosen": extraction["best_transform_chosen"],
            "extracted_text": extraction["extracted_text"],
            "expected_text_normalized": normalize_text(expected_text),
            "extracted_text_normalized": extraction["extracted_text_normalized"],
            "exact_match_normalized": normalize_text(expected_text) == extraction["extracted_text_normalized"],
            "best_avg_confidence": extraction["best_avg_confidence"],
        }

        results.append(result_row)

    results_df = pd.DataFrame(results)

    output_path = ANALYSIS_RESULTS_DIR / "guardrail_extractor_test_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved test results to: {output_path}")

    if not results_df.empty:
        match_rate = results_df["exact_match_normalized"].mean()
        print(f"Guardrail extractor exact match rate: {match_rate:.2%}")

    print()
    print("=== Match Rate by True Variant ===")
    variant_summary = (
        results_df.groupby("true_variant")
        .agg(
            total_cases=("id", "count"),
            matches=("exact_match_normalized", "sum"),
            match_rate=("exact_match_normalized", "mean"),
        )
        .reset_index()
    )
    print(variant_summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)