import os
import pandas as pd
import shutil
from datasets import load_dataset
from .Base_Dataset import Base_Dataset_Class

class Biber_Study_Dataset(Base_Dataset_Class):
    def __init__(self, data_path, cutoffdate=None, n_shot=0, is_test=False):
        super().__init__()
        self.dataset = pd.read_csv(os.path.join(data_path, "Dataset_Subsampled.csv"), index_col=0)
        self.few_shot_dataset = pd.read_csv(os.path.join(data_path, "Dataset_Few_Shot.csv"), index_col=0)
        print(f"Dataset consists of {len(self.dataset)} samples")
        print(f"Load from: {data_path}")
        if is_test:
            self.dataset = self.dataset.iloc[:10]
        self.i = None
        self.n_shot = n_shot

    def __iter__(self):
        self._iter = iter(self.dataset.index)
        return self
    
    def __len__(self):
        return len(self.dataset)
    
    def __next__(self):
        idx = next(self._iter)  # raises StopIteration automatically
        sample = self.dataset.loc[idx]
        if self.n_shot > 0:
            few_shot_samples = self.few_shot_dataset.sample(n=self.n_shot)
        else:
            few_shot_samples = None
        return idx, sample, few_shot_samples
    
    def set_n_shot(self, n_shot):
        self.n_shot = n_shot
