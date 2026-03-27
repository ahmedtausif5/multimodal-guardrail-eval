import sys
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    PROMPTS_DIR,
    CLEAN_IMAGES_DIR,
    MIRROR_IMAGES_DIR,
    ROTATE_IMAGES_DIR,
)


IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 300
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
FONT_SIZE = 32
MARGIN = 40
MAX_LINE_WIDTH = 40  # roughly how many characters per line


def load_font():
    """
    Try to load a nicer font; fall back to default if unavailable.
    """
    possible_fonts = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "Arial.ttf",
    ]

    for font_path in possible_fonts:
        try:
            return ImageFont.truetype(font_path, FONT_SIZE)
        except Exception:
            continue

    return ImageFont.load_default()


def wrap_text(text: str) -> str:
    """
    Wrap long text into multiple lines so it fits better in the image.
    """
    return "\n".join(textwrap.wrap(text, width=MAX_LINE_WIDTH))


def create_base_image(text: str, font) -> Image.Image:
    """
    Create a clean white image with centered wrapped black text.
    """
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    wrapped_text = wrap_text(text)

    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=10)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (IMAGE_WIDTH - text_width) // 2
    y = (IMAGE_HEIGHT - text_height) // 2

    draw.multiline_text(
        (x, y),
        wrapped_text,
        fill=TEXT_COLOR,
        font=font,
        spacing=10,
        align="center",
    )

    return image


def save_variants(prompt_id: str, text: str, font):
    """
    Create and save clean, mirror, and rotated versions of the prompt image.
    Returns metadata rows describing the saved files.
    """
    metadata_rows = []

    clean_image = create_base_image(text, font)

    clean_path = CLEAN_IMAGES_DIR / f"{prompt_id}.png"
    clean_image.save(clean_path)
    metadata_rows.append(
        {
            "id": prompt_id,
            "variant": "clean",
            "image_path": str(clean_path),
            "prompt_text": text,
        }
    )

    mirror_image = clean_image.transpose(Image.FLIP_LEFT_RIGHT)
    mirror_path = MIRROR_IMAGES_DIR / f"{prompt_id}.png"
    mirror_image.save(mirror_path)
    metadata_rows.append(
        {
            "id": prompt_id,
            "variant": "mirror",
            "image_path": str(mirror_path),
            "prompt_text": text,
        }
    )

    rotate_image = clean_image.rotate(180)
    rotate_path = ROTATE_IMAGES_DIR / f"{prompt_id}.png"
    rotate_image.save(rotate_path)
    metadata_rows.append(
        {
            "id": prompt_id,
            "variant": "rotate",
            "image_path": str(rotate_path),
            "prompt_text": text,
        }
    )

    return metadata_rows


def main():
    csv_path = PROMPTS_DIR / "curated_prompts.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Process only enabled rows
    enabled_df = df[df["enabled"] == True].copy()

    if enabled_df.empty:
        raise ValueError("No enabled prompts found in curated_prompts.csv")

    font = load_font()

    all_metadata = []

    print("=== Rendering Enabled Prompts ===")
    for _, row in enabled_df.iterrows():
        prompt_id = row["id"]
        prompt_text = row["prompt_text"]

        print(f"Rendering {prompt_id} ...")
        rows = save_variants(prompt_id, prompt_text, font)
        all_metadata.extend(rows)

    metadata_df = pd.DataFrame(all_metadata)

    metadata_path = PROMPTS_DIR / "rendered_images_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print()
    print("=== Done ===")
    print(f"Rendered {len(enabled_df)} prompts.")
    print(f"Created {len(all_metadata)} images total.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)