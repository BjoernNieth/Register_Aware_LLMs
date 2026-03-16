import hydra
from omegaconf import DictConfig

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import ensure_exists_dirs

@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data", "Dataset.csv")
    EMBEDDED_DATA_PATH = os.path.join(DATASET_PATH, "embedding")
    ensure_exists_dirs([DATASET_PATH, PREPARED_DATA_PATH, EMBEDDED_DATA_PATH])
    dataset = pd.read_csv(os.path.join(PREPARED_DATA_PATH), index_col=0)
    # Load model (small & fast, adjust as needed)
    model = SentenceTransformer(cfg.embedding.hf_name)

    # Compute embeddings
    embeddings = model.encode(
        (dataset[cfg.dataset.task_name] + " " + dataset[cfg.dataset.text_name]).tolist(),
        batch_size=cfg.embedding.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # optional: L2-normalize for cosine similarity
    )

    # Save to NumPy
    np.savez(os.path.join(EMBEDDED_DATA_PATH, cfg.embedding.name + ".npz"), ids=dataset.index.to_list() ,embeddings=embeddings)


if __name__ == "__main__":
    main()