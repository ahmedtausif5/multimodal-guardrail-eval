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
    IMAGES_DIR,
)

IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 360
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
FONT_SIZE = 30
MAX_LINE_WIDTH = 44


def load_font():
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
    return "\n".join(textwrap.wrap(text, width=MAX_LINE_WIDTH))


def create_base_image(text: str, font) -> Image.Image:
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


def main():
    csv_path = PROMPTS_DIR / "controlled_proxy_prompts.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Controlled proxy prompt file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("controlled_proxy_prompts.csv is empty")

    enabled_df = df[df["enabled"] == True].copy()

    if enabled_df.empty:
        raise ValueError("No enabled rows found in controlled_proxy_prompts.csv")

    controlled_root = IMAGES_DIR / "controlled_proxy"
    clean_dir = controlled_root / "clean"
    mirror_dir = controlled_root / "mirror"
    rotate_dir = controlled_root / "rotate"

    clean_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    rotate_dir.mkdir(parents=True, exist_ok=True)

    font = load_font()
    metadata_rows = []

    print("=== Rendering Controlled Proxy Prompts ===")
    print(f"Input file: {csv_path}")
    print(f"Enabled rows: {len(enabled_df)}")
    print()

    for _, row in enabled_df.iterrows():
        prompt_id = row["id"]
        prompt_text = row["prompt_text"]
        category = row["category"]
        expected_guardrail_decision = row["expected_guardrail_decision"]

        print(f"Rendering {prompt_id} ...")

        clean_image = create_base_image(prompt_text, font)

        clean_path = clean_dir / f"{prompt_id}.png"
        clean_image.save(clean_path)
        metadata_rows.append(
            {
                "id": prompt_id,
                "variant": "clean",
                "image_path": str(clean_path),
                "prompt_text": prompt_text,
                "category": category,
                "expected_guardrail_decision": expected_guardrail_decision,
            }
        )

        mirror_image = clean_image.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_path = mirror_dir / f"{prompt_id}.png"
        mirror_image.save(mirror_path)
        metadata_rows.append(
            {
                "id": prompt_id,
                "variant": "mirror",
                "image_path": str(mirror_path),
                "prompt_text": prompt_text,
                "category": category,
                "expected_guardrail_decision": expected_guardrail_decision,
            }
        )

        rotate_image = clean_image.rotate(180)
        rotate_path = rotate_dir / f"{prompt_id}.png"
        rotate_image.save(rotate_path)
        metadata_rows.append(
            {
                "id": prompt_id,
                "variant": "rotate",
                "image_path": str(rotate_path),
                "prompt_text": prompt_text,
                "category": category,
                "expected_guardrail_decision": expected_guardrail_decision,
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)

    output_path = PROMPTS_DIR / "controlled_proxy_rendered_metadata.csv"
    metadata_df.to_csv(output_path, index=False)

    print()
    print("=== Done ===")
    print(f"Rendered prompts: {len(enabled_df)}")
    print(f"Total images created: {len(metadata_df)}")
    print(f"Metadata saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)