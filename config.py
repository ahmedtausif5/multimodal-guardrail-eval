from pathlib import Path

# -----------------------------
# Project root
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------
# Data folders
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = DATA_DIR / "results"

# -----------------------------
# Image subfolders
# -----------------------------
CLEAN_IMAGES_DIR = IMAGES_DIR / "clean"
MIRROR_IMAGES_DIR = IMAGES_DIR / "mirror"
ROTATE_IMAGES_DIR = IMAGES_DIR / "rotate"
TYPOGRAPHY_IMAGES_DIR = IMAGES_DIR / "typography"

# -----------------------------
# Result subfolders
# -----------------------------
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
DEFENDED_RESULTS_DIR = RESULTS_DIR / "defended"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"

# -----------------------------
# Models
# -----------------------------
PRIMARY_MODEL = "gemini-2.5-flash"

# Optional future second model
ENABLE_SECOND_MODEL = False
SECOND_MODEL_NAME = "llama-3.2-11b-vision-instruct"

# -----------------------------
# Experiment safety / control settings
# -----------------------------
MAX_REQUESTS_TOTAL = 100
MAX_REQUESTS_PER_RUN = 20
MAX_OUTPUT_TOKENS = 300
RANDOM_SEED = 42

# -----------------------------
# Utility: create all directories
# -----------------------------
ALL_DIRS = [
    DATA_DIR,
    PROMPTS_DIR,
    IMAGES_DIR,
    RESULTS_DIR,
    CLEAN_IMAGES_DIR,
    MIRROR_IMAGES_DIR,
    ROTATE_IMAGES_DIR,
    TYPOGRAPHY_IMAGES_DIR,
    BASELINE_RESULTS_DIR,
    DEFENDED_RESULTS_DIR,
    ANALYSIS_RESULTS_DIR,
]