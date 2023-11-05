import pandas as pd

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

