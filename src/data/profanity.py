"""
This script downloads list of English profanities and
transforms it into a table with single column.
"""
from pathlib import Path

import pandas as pd
import wget
import typer

from src.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

DOWNLOAD_URL = "https://github.com/surge-ai/profanity/raw/main/profanity_en.csv"


def download_profanity(download_url: str = DOWNLOAD_URL, save_path: Path = EXTERNAL_DATA_DIR,
                       preprocessed_file: Path = INTERIM_DATA_DIR / "bad_words.txt"):
    file_name = wget.download(download_url, out=str(save_path))
    df = pd.read_csv(file_name)
    result = pd.concat(
        [
            df[df.canonical_form_1.str.count(" ") == 0].canonical_form_1,
            df[df.text.str.count(" ") == 0].text,
            df[df.canonical_form_2.str.count(" ") == 0].canonical_form_2,
            df[df.canonical_form_3.str.count(" ") == 0].canonical_form_3,
        ]
    ).unique()
    (
        pd.DataFrame(result).to_csv(
            preprocessed_file, index=False, header=False
        )
    )


if __name__ == '__main__':
    typer.run(download_profanity)
