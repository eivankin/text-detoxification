import gc
import random

import numpy as np
import torch
import transformers


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_avg(avg: float, new_val: float, num_elem: int) -> float:
    return (avg * num_elem + new_val) / (num_elem + 1)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
