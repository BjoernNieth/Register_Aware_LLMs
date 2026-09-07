import random
import numpy as np
import torch
import pandas as pd
import vllm 


def set_random_seed(seed: int = 2701):
    """
    Set random seed for NumPy, PyTorch, Pandas, and other common libraries.

    :param seed: The seed value to use.
    :param deterministic: If True, sets PyTorch to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pd.util.hash_pandas_object = lambda obj: obj.apply(lambda x: hash(tuple(x)), axis=1)

    print(f"Random seed set to {seed}")