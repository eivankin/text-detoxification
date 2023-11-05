from pathlib import Path

import pandas as pd
import typer

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.data.download_paranmt import read_dataset


def swap_translation_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new dataframe by swapping `translation` with `reference` and their scores if `trn_tox` > `ref_tox`
    """
    increased_toxicity_mask = df.trn_tox > df.ref_tox
    increased_toxicity = df[increased_toxicity_mask]
    swapped = increased_toxicity.copy()
    swapped["translation"] = increased_toxicity["reference"]
    swapped["reference"] = increased_toxicity["translation"]
    swapped["trn_tox"] = increased_toxicity["ref_tox"]
    swapped["ref_tox"] = increased_toxicity["trn_tox"]

    result = df.copy()
    result[increased_toxicity_mask] = swapped
    return result


def preprocess_dataset(input_dataframe: Path = RAW_DATA_DIR / "filtered.tsv",
                       output_path: Path = INTERIM_DATA_DIR / "swapped.csv"):
    """This script swaps translation with reference in given dictionary
    if `trn_tox` > `ref_tox` and saves it to given path."""
    df = read_dataset(input_dataframe)
    swap_translation_direction(df).to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(preprocess_dataset)
