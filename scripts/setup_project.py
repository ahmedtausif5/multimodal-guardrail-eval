import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import ALL_DIRS


def main():
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created or already exists: {directory}")


if __name__ == "__main__":
    main()