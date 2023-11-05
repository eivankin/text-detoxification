"""
Utils for transforming pandas dataframe to Dataset object
"""
from typing import Iterable

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


class DeToxDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """Accepts preprocessed dataset"""
        self._df = df.drop(columns=["similarity", "lenght_diff", "ref_tox", "trn_tox"])

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self._df.iloc[idx, :][["reference", "translation"]]


def collate_batch(batch: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    return [item[0] for item in batch], [item[1] for item in batch]


def create_loaders(df: pd.DataFrame, train_fraction: float = 0.8, test_fraction: float = 0.05,
                   batch_size: int = 64) -> \
        tuple[DataLoader, DataLoader, DataLoader]:
    dataset = DeToxDataset(df)
    train_size, test_size = int(len(dataset) * train_fraction), int(len(dataset) * test_fraction)
    val_size = len(dataset) - train_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, (train_size, val_size, test_size))
    return DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                      collate_fn=collate_batch), DataLoader(val_ds, batch_size=batch_size,
                                                            collate_fn=collate_batch), DataLoader(
        test_ds, batch_size=batch_size, collate_fn=collate_batch)
