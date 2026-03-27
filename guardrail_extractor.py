from pathlib import Path

from PIL import Image
import pytesseract


def normalize_text(text: str) -> str:
    """
    Normalize text for easier comparison.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def run_ocr_with_confidence(image: Image.Image):
    """
    Run OCR and estimate OCR quality using average word confidence.

    Returns:
        extracted_text (str)
        avg_confidence (float)
    """
    extracted_text = pytesseract.image_to_string(image)

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DATAFRAME
    )

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
    """
    candidates = {
        "original": image.copy(),
        "flip_horizontal": image.transpose(Image.FLIP_LEFT_RIGHT),
        "rotate_180": image.rotate(180),
    }
    return candidates


def choose_best_candidate(image: Image.Image):
    """
    Try OCR on multiple candidate transforms and choose the best one
    based on OCR confidence, then normalized text length.
    """
    candidates = generate_candidates(image)
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

    candidate_results = sorted(
        candidate_results,
        key=lambda x: (x["avg_confidence"], x["normalized_length"]),
        reverse=True,
    )

    best_candidate = candidate_results[0]
    return best_candidate, candidate_results


def extract_text_with_candidate_normalization(image_path: str | Path):
    """
    Main reusable function for the guardrail extractor.

    Input:
        image_path: path to the image file

    Returns a dictionary with:
        - image_path
        - best_transform_chosen
        - extracted_text
        - extracted_text_normalized
        - best_avg_confidence
        - all_candidate_scores
    """
    image_path = Path(image_path)
    image = Image.open(image_path)

    best_candidate, all_candidates = choose_best_candidate(image)

    return {
        "image_path": str(image_path),
        "best_transform_chosen": best_candidate["transform_name"],
        "extracted_text": best_candidate["ocr_text"],
        "extracted_text_normalized": best_candidate["ocr_text_normalized"],
        "best_avg_confidence": best_candidate["avg_confidence"],
        "all_candidate_scores": [
            {
                "transform_name": c["transform_name"],
                "avg_confidence": c["avg_confidence"],
                "normalized_length": c["normalized_length"],
            }
            for c in all_candidates
        ],
    }