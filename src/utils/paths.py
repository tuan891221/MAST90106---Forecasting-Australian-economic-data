from pathlib import Path

# automatically find project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "project_data"
MERGED_DATA_DIR = DATA_DIR / "data" / "merged"

OUTPUT_DIR = PROJECT_ROOT / "outputs"