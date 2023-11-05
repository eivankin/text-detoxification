"""
This script downloads the filtered ParaNMT-detox corpus (500K sentence pairs)
into the `./data/raw` directory and unpacks it without keeping the source archive.
"""

import os
import zipfile

import wget

from src.config import RAW_DATA_DIR

# From task description
DOWNLOAD_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"

if __name__ == "__main__":
    file_name = wget.download(DOWNLOAD_URL, out=str(RAW_DATA_DIR))

    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

    os.unlink(file_name)
