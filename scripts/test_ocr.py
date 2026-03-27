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


def main():
    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Rendered metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("rendered_images_metadata.csv is empty")

    results = []

    print("=== Running OCR Test on Rendered Images ===")
    print(f"Loaded metadata from: {metadata_path}")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        variant = row["variant"]
        image_path = row["image_path"]
        expected_text = row["prompt_text"]

        print(f"[{i + 1}/{len(df)}] OCR on {prompt_id} | {variant}")

        image = Image.open(image_path)

        # Run OCR
        ocr_output = pytesseract.image_to_string(image)

        result_row = {
            "id": prompt_id,
            "variant": variant,
            "image_path": image_path,
            "expected_text": expected_text,
            "ocr_output": ocr_output,
            "expected_text_normalized": normalize_text(expected_text),
            "ocr_output_normalized": normalize_text(ocr_output),
            "exact_match_normalized": normalize_text(expected_text) == normalize_text(ocr_output),
        }

        results.append(result_row)

    results_df = pd.DataFrame(results)

    output_path = ANALYSIS_RESULTS_DIR / "ocr_test_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved OCR results to: {output_path}")

    if not results_df.empty:
        match_rate = results_df["exact_match_normalized"].mean()
        print(f"OCR normalized exact match rate: {match_rate:.2%}")

    print()
    print("=== OCR Match Rate by Variant ===")
    variant_summary = (
        results_df.groupby("variant")
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