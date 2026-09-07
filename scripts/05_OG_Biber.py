import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
from utils import string_cleaning, coerce_numeric, punctuation_ratio, ensure_exists_dirs, plot_length_hist, count_words, window_text_soft, get_dimensional_loading_1988
import pandas as pd 
import polars as pl
import pybiber as pb
import spacy
import gc
import tqdm
import time



def get_dimensional_loadings(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    EXPERIMENT_PATH = os.path.join(DATASET_PATH, "experiments")
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    
    print("Load Biber loadings")
    ## Load original values of the 1988 Biber study. Remove f_62 as it has zero std and is not used in any loading
    # Drop f_23, as it is not stable from PyBiber
    biber_loadings_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "biber_loadings", "biber_loadings_1988.csv")).drop(["f_62_split_infinitive"], axis=1)
    print(biber_loadings_df)
    biber_normalizations_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "biber_loadings", "biber_normalisations_1988.csv"))
    biber_normalizations_df = biber_normalizations_df[~biber_normalizations_df["Feature"].isin(["f_62_split_infinitive"])]

    ensure_exists_dirs([ARTEFACT_PATH])

    results = []


    human_biber_df_full = pd.read_parquet(os.path.join(PREPARED_DATA_PATH, "dataset_en_core_web_trf_biber.parquet")).set_index("doc_id").sort_index(axis=1).drop(["f_62_split_infinitive"], axis=1)
    result_df = get_dimensional_loading_1988(human_biber_df_full, biber_loadings_df, biber_normalizations_df)
    result_df["group"] = "Human_Full"
    results.append(result_df)
    
    # Split membership is read from the Biber feature tables, not the raw text CSVs, so the
    # pipeline runs without redistributing licensed corpus text (BNC2014Spoken, XSum).
    subsampled_df = pd.read_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled_Biber_Features.csv"), index_col=0)
    human_biber_df = human_biber_df_full.loc[subsampled_df.index.astype(str)]
    result_df = get_dimensional_loading_1988(human_biber_df, biber_loadings_df, biber_normalizations_df)
    result_df["group"] = "Human"
    results.append(result_df)

    few_shot_df = pd.read_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Few_Shot_Biber_Features.csv"), index_col=0)
    human_biber_df_few_shot = human_biber_df_full.loc[few_shot_df.index.astype(str)]
    result_df = get_dimensional_loading_1988(human_biber_df_few_shot, biber_loadings_df, biber_normalizations_df)
    result_df["group"] = "Human_Few_Shot"
    results.append(result_df)

    for experiment in os.scandir(EXPERIMENT_PATH):
        if experiment.is_dir():
            try:
                experiment_df = pd.read_csv(os.path.join(experiment.path, f"dataset_{cfg.biber_tag.spacy_model}_biber.csv"), index_col=0).drop(["f_62_split_infinitive"], axis=1)
                result_df = get_dimensional_loading_1988(experiment_df, biber_loadings_df, biber_normalizations_df)
                result_df["group"] = experiment.name
                results.append(result_df)
            except Exception as e:
                print(f"Skipping {experiment.name}: {type(e).__name__}: {e}")

    results_df = pd.concat(results)
    results_df["dataset"] = cfg.dataset.name

    print(f"Saved results to: {os.path.join(ARTEFACT_PATH, "Dimensional_Loadings_1988.csv")}")
    results_df.to_csv(os.path.join(ARTEFACT_PATH, "Dimensional_Loadings_1988.csv"))




@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")
    print(f"Dataset: {cfg.dataset.name}")
    get_dimensional_loadings(cfg)

if __name__ == "__main__":
    main()



