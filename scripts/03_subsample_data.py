import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
import pandas as pd
from utils import get_dimensional_loading, coerce_numeric, get_dimensional_loading, dimensional_matching_wasserstein, get_tokenized_length
from stats import compute_mmd_unbiased_batched, get_cohens_d
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle 
import tqdm 



def write_stats(DOCUMENTATION_FILE_SUBAMPLING, X_scaled, biber_df, ids_subsampled, cfg, lowest_score, biber_dimensions, ids, cutt_off):
    X_subsampled_scaled = X_scaled[ids_subsampled]
    biber_dimensions_subsampled = biber_dimensions.iloc[ids_subsampled]
    mmd = compute_mmd_unbiased_batched(X_scaled, X_subsampled_scaled)
    cohens_ds = []
    with open(DOCUMENTATION_FILE_SUBAMPLING, "w") as f:
        f.write(f"Selected {cfg.subsample.sample_size} samples out of {len(ids)} samples with {X_scaled.shape[0]} samples total in dataset.\n")
        f.write(f"The cutoff for the context data was {cutt_off} tokens, so every sample has in worst case {cutt_off} tokens as context.\n")
        f.write(f"Global squared unbiased MMD between subsample and sample: {mmd}\n")
        f.write("-----------------------------------------------------------------------------------------\n")
        f.write("| Feature            | Mean_Full | Mean_Subsample | STD_Full | STD_Subsample | Cohens D |\n")
        f.write("-----------------------------------------------------------------------------------------\n")

        unsorted_columns = list(biber_df.columns)
        sorted_columns = list(biber_df.columns)
        sorted_columns.sort()
        for col in sorted_columns:
            column_id = unsorted_columns.index(col)
            mean_full = X_scaled[:, column_id].mean()
            mean_subsample = X_subsampled_scaled[:, column_id].mean()
            std_full = X_scaled[:, column_id].std()
            std_subsample = X_subsampled_scaled[:, column_id].std()

            cohens_d = get_cohens_d(mean_full, mean_subsample, std_full, std_subsample, X_scaled.shape[0], X_subsampled_scaled.shape[0])
            cohens_ds.append(cohens_d)
            f.write(
                f"| {col.ljust(20)[:20]}| {mean_full:10.4f} | {mean_subsample:14.4f} | "
                f"{std_full:10.4f} | {std_subsample:13.4f} | {cohens_d:10.3f} |\n"
            )   
        f.write("-----------------------------------------------------------------------------------------\n")
        f.write("-----------------------------------------------------------------------------------------\n")
        f.write("-----------------------------------------------------------------------------------------\n")
        f.write("-----------------------------------------------------------------------------------------\n")
        f.write("| Dimension            | Mean_Full | Mean_Subsample | STD_Full | STD_Subsample |\n")
        f.write("-----------------------------------------------------------------------------------------\n")

        dim = 1
        for mean_full, mean_sub, std_full, std_sub in zip(biber_dimensions.mean(), biber_dimensions_subsampled.mean(),
                                                        biber_dimensions.std(), biber_dimensions_subsampled.std()):
            f.write(
                f"| {f"Dimension_{dim}".ljust(20)[:20]}| {mean_full:10.4f} | {mean_sub:14.4f} | "
                f"{std_full:10.4f} | {std_sub:13.4f} |\n"
            )   
            dim += 1
        f.write("-----------------------------------------------------------------------------------------\n")

        f.write(f"Lowest mean Wasserstein distance: {lowest_score}\n")
        f.write(f"Gloabel cohens_d: {np.abs(np.array(cohens_d)).mean()}\n")
        f.write(f"Selected ids:\n {ids_subsampled}")





def dimensional_matching(biber_df_full, biber_df_subsampled):
    eps = np.finfo(float).eps
    full_dimensions = biber_df_full.values
    subsampled_dimension = biber_df_subsampled.values
    # Full corpus stats per dimension
    full_mean = full_dimensions.mean(axis=0)
    full_std  = full_dimensions.std(axis=0, ddof=1)
    
    # Subsample stats per dimension
    subsampled_mean = subsampled_dimension.mean(axis=0)
    subsampled_std  = subsampled_dimension.std(axis=0, ddof=1)
    
    # Relative deviations (scale-free)
    delta_mean = np.abs(subsampled_mean - full_mean) / (np.abs(full_mean) + eps)
    delta_std  = np.abs(subsampled_std - full_std)   / (np.abs(full_std) + eps)
    
    # Combine mean + std deviations per dimension
    S_d = delta_mean + delta_std
    
    # Aggregate across dimensions
    score = S_d.mean()
    return float(score)

def subsample_dataset(cfg):
    np.random.seed(cfg.run.seed)
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    DOCUMENTATION_FILE_SUBAMPLING = os.path.join(ARTEFACT_PATH, "subsample.txt")
    DOCUMENTATION_FILE_FEWSHOT = os.path.join(ARTEFACT_PATH, "few_shot.txt")

    print("Load datasets")
    original_dataset = pd.read_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"), index_col=0)
    original_dataset.index = original_dataset.index.astype(str)
    original_dataset = original_dataset.sort_index()
    
    biber_df = pd.read_parquet(os.path.join(PREPARED_DATA_PATH, f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet")).set_index("doc_id")
    biber_df.index = biber_df.index.astype(str)
    biber_df = biber_df.sort_index()

    # Ensure the ids are aligned
    assert original_dataset.index.equals(biber_df.index), "Different index for biber features and dataset"

    # Convert to float64 for more stability 
    biber_df = coerce_numeric(biber_df)
    
    print(f"Get {cfg.dataset.percentile_cutoff} percentile cutoff for context data length tokenized by Llama model.")
    tokenized_length = get_tokenized_length(original_dataset, cfg.dataset.context_data_columns, cfg.run.hf_token)
    cutt_off = np.percentile(tokenized_length, cfg.dataset.percentile_cutoff)
    ids = np.array(range(len(biber_df)))

    # Only keep the ids for subsampling without very large metadata
    ids = ids[tokenized_length < cutt_off]
    print(f"Dataset for subsampling now consists of {len(ids)}/{len(biber_df)} samples.")
    print("Z-scale biber features")
    # Fit scaler on full dataset
    scaler = StandardScaler()
    biber_dimensions = get_dimensional_loading(biber_df)
    X = biber_df.values
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    print(f"Printed subsampling documentation to: {DOCUMENTATION_FILE_SUBAMPLING}")
    lowest_score = 100000
    ids_subsampled_best = None
    for _ in tqdm.tqdm(range(cfg.subsample.random_draws)):
        # Get the subsample set for the experiments
        ids_subsampled = np.random.choice(ids, size=cfg.subsample.sample_size, replace=False)
        score = dimensional_matching_wasserstein(biber_dimensions, biber_dimensions.iloc[ids_subsampled])
        if score < lowest_score:
            ids_subsampled_best = ids_subsampled.copy()
            lowest_score = score

    assert ids_subsampled_best is not None
    write_stats(DOCUMENTATION_FILE_SUBAMPLING, X_scaled, biber_df, ids_subsampled_best, cfg, lowest_score, biber_dimensions, ids, cutt_off)
    
    print(f"Printed few-shot documentation to: {DOCUMENTATION_FILE_SUBAMPLING}")
    # Get the subsample set for the few-shot examples
    lowest_score = 100000
    ids_subsampled_best_2 = None
    ids_not_used = [i for i in ids if i not in ids_subsampled_best]
    for _ in range(cfg.subsample.random_draws):
        # Get the subsample set for the experiments
        ids_subsampled = np.random.choice(ids_not_used, size=cfg.subsample.sample_size, replace=False)
        score = dimensional_matching_wasserstein(biber_dimensions, biber_dimensions.iloc[ids_subsampled])
        if score < lowest_score:
            ids_subsampled_best_2 = ids_subsampled.copy()
            lowest_score = score
    assert ids_subsampled_best_2 is not None
    write_stats(DOCUMENTATION_FILE_FEWSHOT, X_scaled, biber_df, ids_subsampled_best_2, cfg, lowest_score, biber_dimensions, ids, cutt_off)

    print(f"Print subsampled dataset to {PREPARED_DATA_PATH}")
    subsampled_dataset_original = original_dataset.iloc[ids_subsampled_best]
    subsampled_dataset_original.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled.csv"))

    print(f"Print rest of dataset to {os.path.join(PREPARED_DATA_PATH, "Dataset_Few_Shot.csv")}")
    few_shot_dataset_original = original_dataset.iloc[ids_subsampled_best_2]
    few_shot_dataset_original.to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Few_Shot.csv"))

    print(f"Subsampled Biberfeatures to {os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled_Biber_Features.csv")}")
    biber_df.iloc[ids_subsampled_best].sort_index(axis=1).to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled_Biber_Features.csv"))
    biber_df.iloc[ids_subsampled_best_2].sort_index(axis=1).to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Few_Shot_Biber_Features.csv"))

    #print(f"Subsampled Biberfeatures normalized on full dataset to {os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled_Biber_Features_z_score.csv")}")
    #biber_df_z_scores = biber_df.copy()
    #biber_df_z_scores.loc[:, :] = scaler.transform(biber_df_z_scores)
    #biber_df_z_scores.iloc[ids_subsampled_best].sort_index(axis=1).to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled_Biber_Features_z_score.csv"))
    #biber_df_z_scores.iloc[ids_subsampled_best_2].sort_index(axis=1).to_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Few_Shot_Biber_Features_z_score.csv"))
    

@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")

    print(f"Dataset: {cfg.dataset.name}")
    subsample_dataset(cfg)


if __name__ == "__main__":
    main()



