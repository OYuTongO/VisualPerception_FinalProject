# Helper script: download ASL Alphabet dataset from Kaggle
#
# Usage:
#   1. Place kaggle.json in ~/.kaggle/  (get it from kaggle.com -> Account -> API)
#   2. pip install kaggle
#   3. python scripts/download_dataset.py

import subprocess
import zipfile
import os

DATASET = "grassknoted/asl-alphabet"
OUTPUT_DIR = "data/raw"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Downloading dataset...")
    subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "--path", OUTPUT_DIR],
        check=True,
    )
    zip_path = os.path.join(OUTPUT_DIR, "asl-alphabet.zip")
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(OUTPUT_DIR)
    os.remove(zip_path)
    print(f"Done. Dataset saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
