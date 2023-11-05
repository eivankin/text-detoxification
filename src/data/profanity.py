"""
This script downloads list of English profanities and
transforms it into a table with single column.
"""

import wget
import pandas as pd

from src.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

DOWNLOAD_URL = "https://github.com/surge-ai/profanity/raw/main/profanity_en.csv"

if __name__ == '__main__':
    file_name = wget.download(DOWNLOAD_URL, out=str(EXTERNAL_DATA_DIR))
    df = pd.read_csv(file_name)
    result = pd.concat([df[df.canonical_form_1.str.count(' ') == 0].canonical_form_1,
                        df[df.text.str.count(' ') == 0].text,
                        df[df.canonical_form_2.str.count(' ') == 0].canonical_form_2,
                        df[df.canonical_form_3.str.count(' ') == 0].canonical_form_3]).unique()
    (pd.DataFrame(result)
     .to_csv(INTERIM_DATA_DIR / "bad_words.txt", index=False, header=False))
