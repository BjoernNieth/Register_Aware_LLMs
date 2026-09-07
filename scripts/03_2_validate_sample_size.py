import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
import pandas as pd
from utils import get_dimensional_loading, coerce_numeric, get_dimensional_loading, dimensional_matching_wasserstein
from stats import compute_mmd_unbiased_batched, get_cohens_d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import tqdm 




def validate_sample_size(cfg):
    np.random.seed(cfg.run.seed)
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_SUBAMPLING = os.path.join(ARTEFACT_PATH, "subsample.txt")
    DOCUMENTATION_FILE_FEWSHOT = os.path.join(ARTEFACT_PATH, "few_shot.txt")

    print("Load datasets")
    #original_dataset = pd.read_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"), index_col=0)
    biber_df = pd.read_parquet(os.path.join(PREPARED_DATA_PATH, f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet")).set_index("doc_id")
    # Convert to float64 for more stability 
    biber_df = coerce_numeric(biber_df)

    print("Z-scale biber features")
    # Fit scaler on full dataset
    scaler = StandardScaler()
    biber_dimensions = get_dimensional_loading(biber_df)
    X = biber_df.values
    


    print("Run stability analysis")
    rows = []
    scores_dict = {}
    ids = list(range(X.shape[0]))
    for n in tqdm.tqdm(range(cfg.subsample.sample_size_lower, cfg.subsample.sample_size_upper + 1, cfg.subsample.sample_size_step_size)):
        scores = []
        for _ in range(cfg.subsample.sample_size_repetitions):
            # Get the subsample set for the experiments
            ids_subsampled = np.random.choice(ids, size=n, replace=False)
            scores.append(dimensional_matching_wasserstein(biber_dimensions, biber_dimensions.iloc[ids_subsampled]))
        scores_np = np.array(scores)
        scores_dict[str(n)] = scores_np
        # Compute percentile CI
        ci = cfg.stat_eval.CI
        lower = np.percentile(scores_np, 0 + (1-ci)/2)
        upper = np.percentile(scores_np, 1 - (1-ci)/2)

        rows.append({
            "n": n,
            "mean_score": scores_np.mean(),
            "ci_lower": lower,
            "ci_upper": upper,
        })

    print("Save Results")
    stability_curve_df = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    stability_curve_df.to_csv(os.path.join(ARTEFACT_PATH, "stability_curve.csv"))

    np.savez(os.path.join(ARTEFACT_PATH, "stability_curve_values.npz"), **scores_dict)
    print("Visualize Results")
    # Extract columns
    n = stability_curve_df["n"].to_numpy()
    m = stability_curve_df["mean_score"].to_numpy()
    lower = stability_curve_df["ci_lower"].to_numpy()
    upper = stability_curve_df["ci_upper"].to_numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n, m, "o-", label="Mean distance")
    ax.fill_between(n, lower, upper, alpha=0.2, label=f"{int(ci*100)}% CI")

    ax.set_xlabel("Subsample size n")
    ax.set_ylabel("Wasserstein distance (full vs. subsample)")
    ax.set_title(f"Stability of human distribution vs. subsample size ({cfg.dataset.name})")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(os.path.join(ARTEFACT_PATH, "stability_plot.PDF"))
    plt.savefig(os.path.join(ARTEFACT_PATH, "stability_plot.png"))


@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")

    print(f"Dataset: {cfg.dataset.name}")
    validate_sample_size(cfg)


if __name__ == "__main__":
    main()



