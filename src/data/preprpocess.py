import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_df", type=Path, default=RAW_DATA_DIR / "filtered.tsv")
    parser.add_argument(
        "output_path", type=Path, default=INTERIM_DATA_DIR / "swapped.csv"
    )

    args = parser.parse_args()
    df = read_dataset(args.input_df)
    swap_translation_direction(df).to_csv(args.output_path, index=False)
