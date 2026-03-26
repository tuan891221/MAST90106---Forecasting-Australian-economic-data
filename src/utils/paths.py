from pathlib import Path

# automatically find project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"

OUTPUT_DIR = PROJECT_ROOT / "outputs"