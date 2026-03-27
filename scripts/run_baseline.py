import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import os
from google import genai

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    PROMPTS_DIR,
    BASELINE_RESULTS_DIR,
    PRIMARY_MODEL,
    MAX_REQUESTS_PER_RUN,
)

REQUEST_DELAY_SECONDS = 1
RATE_LIMIT_RETRY_SECONDS = 10
MAX_RETRIES = 3


def normalize_text(text: str) -> str:
    """
    Normalize text for easier comparison.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def call_gemini_with_retry(client, model_name, image):
    """
    Call Gemini with basic retry logic for free-tier rate limits.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    "Read the text in this image exactly. Return only the text.",
                    image,
                ],
            )
            return response

        except Exception as e:
            error_text = str(e)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                if attempt < MAX_RETRIES:
                    print(
                        f"Rate limit hit on attempt {attempt}/{MAX_RETRIES}. "
                        f"Sleeping for {RATE_LIMIT_RETRY_SECONDS} seconds before retrying..."
                    )
                    time.sleep(RATE_LIMIT_RETRY_SECONDS)
                else:
                    raise RuntimeError(
                        f"Rate limit persisted after {MAX_RETRIES} attempts: {e}"
                    )
            else:
                raise


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Make sure your .env file contains GEMINI_API_KEY=..."
        )

    client = genai.Client(api_key=api_key)

    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Rendered image metadata file not found: {metadata_path}"
        )

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("rendered_images_metadata.csv is empty")

    # Safety limit so you do not accidentally run too many requests
    df = df.head(MAX_REQUESTS_PER_RUN)

    results = []

    print("=== Running Baseline Evaluation ===")
    print(f"Model: {PRIMARY_MODEL}")
    print(f"Number of image cases to process: {len(df)}")
    print(f"Delay between requests: {REQUEST_DELAY_SECONDS} seconds")
    print()

    for i, row in df.iterrows():
        prompt_id = row["id"]
        variant = row["variant"]
        image_path = row["image_path"]
        expected_text = row["prompt_text"]

        print(f"[{i + 1}/{len(df)}] Processing {prompt_id} | {variant}")

        image = Image.open(image_path)

        response = call_gemini_with_retry(client, PRIMARY_MODEL, image)
        model_output = response.text if response.text else ""

        result_row = {
            "id": prompt_id,
            "variant": variant,
            "image_path": image_path,
            "expected_text": expected_text,
            "model_output": model_output,
            "expected_text_normalized": normalize_text(expected_text),
            "model_output_normalized": normalize_text(model_output),
            "exact_match_normalized": normalize_text(expected_text) == normalize_text(model_output),
            "model_name": PRIMARY_MODEL,
        }

        results.append(result_row)

        # Delay between requests to stay under free-tier rate limits
        if i < len(df) - 1:
            print(f"Sleeping for {REQUEST_DELAY_SECONDS} seconds...")
            time.sleep(REQUEST_DELAY_SECONDS)

    results_df = pd.DataFrame(results)

    output_path = BASELINE_RESULTS_DIR / "baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Saved results to: {output_path}")
    print()

    if not results_df.empty:
        match_rate = results_df["exact_match_normalized"].mean()
        print(f"Normalized exact match rate: {match_rate:.2%}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)