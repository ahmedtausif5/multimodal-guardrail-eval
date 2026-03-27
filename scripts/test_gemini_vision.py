
from dotenv import load_dotenv
import os
from google import genai
from PIL import Image, ImageDraw, ImageFont


def create_test_image(output_path: str) -> None:
    """
    Create a simple white image with black text in the center.
    """
    width, height = 800, 200
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    text = "The hidden word is: pineapple"

    try:
        font = ImageFont.truetype("Arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    # Get text bounding box so we can center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill="black", font=font)
    image.save(output_path)


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Make sure your .env file contains GEMINI_API_KEY=..."
        )

    client = genai.Client(api_key=api_key)

    # Make sure the output folder exists
    os.makedirs("data/images", exist_ok=True)

    image_path = "data/images/test_text_image.png"

    # Create a harmless test image locally
    create_test_image(image_path)

    # Open it as a PIL image
    image = Image.open(image_path)

    # Ask Gemini to read the text in the image
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Read the text in this image exactly. Return only the text.",
            image
        ]
    )

    print("=== Image saved to ===")
    print(image_path)
    print()
    print("=== Gemini Response ===")
    print(response.text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)