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


def apply_oracle_normalization(image: Image.Image, variant: str) -> Image.Image:
    """
    Undo the known transformation based on the metadata label.
    This is an 'oracle' setup for testing whether normalization helps OCR.
    """
    if variant == "clean":
        return image.copy()
    elif variant == "mirror":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif variant == "rotate":
        return image.rotate(180)
    else:
        return image.copy()


def main():
    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Rendered metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("rendered_images_metadata.csv is empty")

    preview_dir = ANALYSIS_RESULTS_DIR / "normalized_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print("=== Running OCR Test with Oracle Normalization ===")
    print(f"Loaded metadata from: {metadata_path}")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        variant = row["variant"]
        image_path = row["image_path"]
        expected_text = row["prompt_text"]

        print(f"[{i + 1}/{len(df)}] Normalizing + OCR on {prompt_id} | {variant}")

        image = Image.open(image_path)
        normalized_image = apply_oracle_normalization(image, variant)

        # Save preview image so you can visually inspect what the normalization did
        preview_path = preview_dir / f"{prompt_id}_{variant}_normalized.png"
        normalized_image.save(preview_path)

        # OCR after normalization
        ocr_output = pytesseract.image_to_string(normalized_image)

        result_row = {
            "id": prompt_id,
            "variant": variant,
            "original_image_path": image_path,
            "normalized_image_path": str(preview_path),
            "expected_text": expected_text,
            "ocr_output": ocr_output,
            "expected_text_normalized": normalize_text(expected_text),
            "ocr_output_normalized": normalize_text(ocr_output),
            "exact_match_normalized": normalize_text(expected_text) == normalize_text(ocr_output),
        }

        results.append(result_row)

    results_df = pd.DataFrame(results)

    output_path = ANALYSIS_RESULTS_DIR / "ocr_oracle_normalized_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved OCR results to: {output_path}")

    if not results_df.empty:
        match_rate = results_df["exact_match_normalized"].mean()
        print(f"OCR normalized exact match rate after oracle normalization: {match_rate:.2%}")

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

    variant_summary_path = ANALYSIS_RESULTS_DIR / "ocr_oracle_normalized_variant_summary.csv"
    variant_summary.to_csv(variant_summary_path, index=False)

    print()
    print("=== Saved Analysis Files ===")
    print(f"Detailed OCR results: {output_path}")
    print(f"Variant summary: {variant_summary_path}")
    print(f"Preview images folder: {preview_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)