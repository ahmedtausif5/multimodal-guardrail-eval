import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import pytesseract

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import PROMPTS_DIR, ANALYSIS_RESULTS_DIR


def normalize_text(text: str) -> str:
    """
    Normalize text for easier comparison.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def run_ocr_with_confidence(image: Image.Image):
    """
    Run OCR and also estimate the OCR quality using average word confidence.
    Returns:
        extracted_text (str)
        avg_confidence (float)
    """
    extracted_text = pytesseract.image_to_string(image)

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DATAFRAME
    )

    # Keep only rows with valid confidence scores
    if "conf" in data.columns:
        valid_conf = data["conf"].dropna()
        valid_conf = valid_conf[valid_conf != -1]
        if len(valid_conf) > 0:
            avg_confidence = float(valid_conf.mean())
        else:
            avg_confidence = -1.0
    else:
        avg_confidence = -1.0

    return extracted_text, avg_confidence


def generate_candidates(image: Image.Image):
    """
    Generate candidate normalized versions of the image.
    We do not use the known variant label here.
    """
    candidates = {
        "original": image.copy(),
        "flip_horizontal": image.transpose(Image.FLIP_LEFT_RIGHT),
        "rotate_180": image.rotate(180),
    }
    return candidates


def choose_best_candidate(candidates: dict):
    """
    OCR each candidate and choose the one with the highest average OCR confidence.
    If confidence ties, prefer the one with longer normalized text.
    """
    candidate_results = []

    for transform_name, candidate_image in candidates.items():
        ocr_text, avg_conf = run_ocr_with_confidence(candidate_image)
        normalized_ocr_text = normalize_text(ocr_text)

        candidate_results.append(
            {
                "transform_name": transform_name,
                "ocr_text": ocr_text,
                "ocr_text_normalized": normalized_ocr_text,
                "avg_confidence": avg_conf,
                "normalized_length": len(normalized_ocr_text),
                "image": candidate_image,
            }
        )

    # Sort by confidence first, then by normalized text length
    candidate_results = sorted(
        candidate_results,
        key=lambda x: (x["avg_confidence"], x["normalized_length"]),
        reverse=True,
    )

    return candidate_results[0], candidate_results


def main():
    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Rendered metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("rendered_images_metadata.csv is empty")

    preview_dir = ANALYSIS_RESULTS_DIR / "candidate_normalized_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("=== Running OCR with Candidate Normalization ===")
    print(f"Loaded metadata from: {metadata_path}")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        true_variant = row["variant"]
        image_path = row["image_path"]
        expected_text = row["prompt_text"]

        print(f"[{i + 1}/{len(df)}] Candidate normalization + OCR on {prompt_id} | {true_variant}")

        image = Image.open(image_path)

        candidates = generate_candidates(image)
        best_candidate, all_candidates = choose_best_candidate(candidates)

        # Save chosen preview image
        preview_path = preview_dir / f"{prompt_id}_{true_variant}_best_{best_candidate['transform_name']}.png"
        best_candidate["image"].save(preview_path)

        result_row = {
            "id": prompt_id,
            "true_variant": true_variant,
            "original_image_path": image_path,
            "best_transform_chosen": best_candidate["transform_name"],
            "best_transform_preview_path": str(preview_path),
            "expected_text": expected_text,
            "ocr_output": best_candidate["ocr_text"],
            "expected_text_normalized": normalize_text(expected_text),
            "ocr_output_normalized": best_candidate["ocr_text_normalized"],
            "exact_match_normalized": normalize_text(expected_text) == best_candidate["ocr_text_normalized"],
            "best_avg_confidence": best_candidate["avg_confidence"],
        }

        results.append(result_row)

    results_df = pd.DataFrame(results)

    output_path = ANALYSIS_RESULTS_DIR / "ocr_candidate_normalized_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved results to: {output_path}")

    if not results_df.empty:
        match_rate = results_df["exact_match_normalized"].mean()
        print(f"Candidate-normalized OCR exact match rate: {match_rate:.2%}")

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

    variant_summary_path = ANALYSIS_RESULTS_DIR / "ocr_candidate_normalized_variant_summary.csv"
    variant_summary.to_csv(variant_summary_path, index=False)

    print()
    print("=== Transform Choices ===")
    transform_counts = (
        results_df.groupby(["true_variant", "best_transform_chosen"])
        .size()
        .reset_index(name="count")
    )
    print(transform_counts)

    transform_counts_path = ANALYSIS_RESULTS_DIR / "ocr_candidate_transform_choices.csv"
    transform_counts.to_csv(transform_counts_path, index=False)

    print()
    print("=== Saved Analysis Files ===")
    print(f"Detailed results: {output_path}")
    print(f"Variant summary: {variant_summary_path}")
    print(f"Transform choices: {transform_counts_path}")
    print(f"Preview images folder: {preview_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)