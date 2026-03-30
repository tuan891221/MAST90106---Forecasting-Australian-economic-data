import requests
from pathlib import Path

try:
    from paths import RAW_DIR
except ImportError:
    RAW_DIR = Path("data/raw")

RAW_DIR.mkdir(parents=True, exist_ok=True)

# RBA G1 inflation historical data
URL = "https://www.rba.gov.au/statistics/tables/xls/g01hist.xlsx"

OUTPUT_PATH = RAW_DIR / "g01hist.xlsx"


def download_file(url, output_path):
    print(f"Downloading from {url} ...")

    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    download_file(URL, OUTPUT_PATH)