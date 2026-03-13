from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Historical training window
START_YEAR = 2025
END_YEAR = 2026

# Results sessions used for historical priors
SESSION_CODES = ["Q", "R", "SQ", "S"]

# Current season logic
CURRENT_SEASON = 2026
REG_RESET_YEAR = 2026

MONTE_CARLO_SIMS = 10000
MIN_EDGE = 0.03
MIN_EV = 0.02
KELLY_FRACTION = 0.25

# Historical prior fades quickly in early 2026
EARLY_SEASON_PRIOR_WEIGHTS = {
    1: 0.30,
    2: 0.25,
    3: 0.20,
    4: 0.15,
    5: 0.10
}
DEFAULT_PRIOR_WEIGHT = 0.05

# Default target weekend for manual runs
DEFAULT_PREDICT_YEAR = 2026
DEFAULT_PREDICT_ROUND = 1

# Ensure directories exist
for folder in [
    DATA_DIR, CACHE_DIR, RAW_DIR, PROCESSED_DIR, METADATA_DIR,
    MODELS_DIR, OUTPUTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)