import os
import pandas as pd
import shutil
from datasets import load_dataset
from .Base_Dataset import Base_Dataset_Class

class Narrative_Beginnings_En(Base_Dataset_Class):
    def __init__(self, data_path, cutoffdate, n_shot=0, is_test=False):
        super().__init__()
        self.dataset = pd.read_csv(os.path.join(data_path, "gutenberg_en_sample_1000_clean.csv"))
        print(f"Dataset consists of {len(self.dataset)} samples")
        self.dataset = self.dataset.rename(columns={"id": "document_id", "genre": "genre_3"}).set_index("document_id")
        self.dataset.index = self.dataset.index.astype(str)
        if is_test:
            self.dataset = self.dataset.iloc[:10]
        self.i = None
        self.name= "Narrative_Beginnings_En"
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
        return str(idx), sample, few_shot_samples
    
    def set_n_shot(self, n_shot):
        self.n_shot = n_shot