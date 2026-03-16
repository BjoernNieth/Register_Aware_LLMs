import os
import pandas as pd
import shutil
from datasets import load_dataset
from .Base_Dataset import Base_Dataset_Class

class WritingPrompts(Base_Dataset_Class):
    def __init__(self, data_path, cutoffdate, n_shot=0, is_test=False):
        super().__init__()
        self.dataset = pd.read_csv(os.path.join(data_path, "WritingPrompts_Cleaned_Counts_Subsampled.csv"), index_col=0)
        print(f"Dataset consists of {len(self.dataset)} samples")
        if is_test:
            self.dataset = self.dataset.iloc[:10]
        self.i = None
        self.name= "WritingPrompts"
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
            few_shot_samples = self.dataset.drop(idx).sample(n=self.n_shot)
        else:
            few_shot_samples = None
        return idx, sample, few_shot_samples
    
    def set_n_shot(self, n_shot):
        self.n_shot = n_shot