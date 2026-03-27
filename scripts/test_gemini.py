from dotenv import load_dotenv
import os
from google import genai


def main():
    # Load environment variables from .env
    load_dotenv()

    # Read the Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Make sure your .env file exists and contains GEMINI_API_KEY=..."
        )

    # Create the Gemini client
    client = genai.Client(api_key=api_key)

    # Send a very simple harmless prompt
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one short sentence."
    )

    print("=== Gemini Response ===")
    print(response.text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__, ":", e)