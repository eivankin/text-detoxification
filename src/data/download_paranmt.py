import os
import zipfile
from pathlib import Path

import pandas as pd
import wget
import typer

from src.config import RAW_DATA_DIR

# From task description
DOWNLOAD_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"


def read_dataset(input_path: Path) -> pd.DataFrame:
    """Reads ParaNMT-detox corpus into pandas dataframe and sets correct index column"""
    return pd.read_csv(input_path, sep="\t", index_col=(0,))

def download_dataset(download_url: str = DOWNLOAD_URL, save_to_dir: Path = RAW_DATA_DIR) -> None:
    """
    This script downloads the filtered ParaNMT-detox corpus (500K sentence pairs)
    into the specified directory (`./data/raw` by default) and unpacks it without keeping the source archive.
    """
    file_name = wget.download(download_url, out=str(save_to_dir))

    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(save_to_dir)

    os.unlink(file_name)


if __name__ == "__main__":
    typer.run(download_dataset)
