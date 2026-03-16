import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
from utils import string_cleaning, punctuation_ratio, ensure_exists_dirs, plot_length_hist, count_words, window_text_soft
import pandas as pd 
import polars as pl
import pybiber as pb
import spacy
import gc
import tqdm
import time
import gc

def biber_tag_experiments(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    EXPERIMENT_PATH = os.path.join(DATASET_PATH, "experiments")
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    spacy.require_gpu()

    print("Setting up the Biber tagging script...")
    print("Loading spaCy model...")
    nlp = spacy.load(cfg.biber_tag.spacy_model, disable=["ner"])

    print()
    experiments = []
    for experiment in os.scandir(EXPERIMENT_PATH):
        if experiment.is_dir():
            if os.path.isfile(os.path.join(experiment.path, f"Model_Output_{cfg.run.min_len_tokens}_words.csv")):
                experiments.append(experiment)
                continue
            try:
                dataset = pd.read_csv(os.path.join(experiment.path, "Model_Output.csv"), index_col=0)
                # For the BNC2014Spoken clear all the speaker tags and actions in the dialogs
                if cfg.dataset.name == "BNC2014Spoken":
                    dataset['generated_text'] = (
                        dataset['generated_text']
                        .str.replace(r"\([^)]*\)", "", regex=True)         # remove (tags)
                        .str.replace(r"\s+", " ", regex=True)
                        # Begin the LLM answer at the first occurance of Speaker_d      
                        .str.replace(r"^.*?(Speaker_\d+)", r"\1", regex=True)
                        .str.replace(r"Speaker_\d+:\s*", "", regex=True)   # remove Speaker_x:
                        .str.replace(r"\s+", " ", regex=True)              # collapse extra spaces
                        .str.strip()
                    )

                dataset["generated_text"] = (
                    dataset["generated_text"]
                    # Remove the thinking taggs 
                    .str.replace(r"<think>.*?</think>", "", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    # Some models enter a lot of "*" into the text
                    .str.replace(r"\*", " ", regex=True)
                    .str.strip())

                n = len(dataset)
                word_count = dataset["generated_text"].apply(count_words)
                #dataset = dataset[word_count >= cfg.run.min_len_tokens]
                if not n == len(dataset):
                    print(f"For {experiment} only {len(dataset)}/{n} samples are longer than {cfg.run.min_len_tokens} words.")
                    raise Exception
                if not len(dataset[word_count >= cfg.run.min_len_tokens]) == cfg.subsample.sample_size:
                    print(f"Experiment: {experiment} only has {len(dataset)} out of the required{n} samples.")
                    print(word_count[word_count < cfg.run.min_len_tokens])
                    #raise Exception
                

                dataset["generated_text"] = dataset["generated_text"].apply(window_text_soft, args=(cfg.run.min_len_tokens, cfg.run.token_cap))
                dataset.to_csv(os.path.join(experiment.path, f"Model_Output_{cfg.run.min_len_tokens}_words.csv"))
                
                experiments.append(experiment)
            except:
                print(f"Skipping {experiment}")

    for experiment in experiments:
        if os.path.isfile(os.path.join(experiment.path, f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet")):
            print(f"Skipping {experiment} as it is already tagged")
            continue

        print(f"Tagging Experiment {experiment.name}")
        print(os.path.join(experiment.path, "Model_Output.csv"))
        dataset = pl.read_csv(os.path.join(experiment.path, f"Model_Output_{cfg.run.min_len_tokens}_words.csv"), has_header=True)
        dataset = dataset.rename({dataset.columns[0]: "doc_id", dataset.columns[1]: "text"})
        biber_df = dataset.select(
            pl.col("doc_id").cast(pl.String),
            pl.col("text")
        )
        assert biber_df.filter(pl.col("text").is_not_null()).height == biber_df.height
        
        # Sort by text length to speed up batch processing
        #biber_df = biber_df.with_columns(
        #        pl.col("text").str.len_chars().alias("text_length")
        #    ).sort("text_length", descending=True)
        #biber_df = biber_df.drop("text_length")


        # Store intermideate results as a ndjson
        biber_output_path_ndjson = os.path.join(CACHE_PATH,  experiment.name + f"_{cfg.biber_tag.spacy_model}_biber.ndjson")
        if os.path.isfile(biber_output_path_ndjson):
            initital_len = biber_df.height
            # Remove all the already tagged texts from the df
            biber_df = biber_df.filter(~biber_df["doc_id"].is_in(pl.read_ndjson(biber_output_path_ndjson)["doc_id"]))
            print(f"{initital_len - biber_df.height} Samples were already Biber tagged!")

        if  biber_df.height > 0:
            print(f"Starting spaCy parsing and Biber Tagging for {experiment.name}")
            for i in tqdm.tqdm(range(0, biber_df.height, cfg.biber_tag.batch_size)):
                # Reinnit processor
                processor = pb.CorpusProcessor()
                gc.collect()
                batch_df = biber_df[i:i+cfg.biber_tag.batch_size]
                df_tokens = processor.process_corpus(
                    batch_df, 
                    nlp_model=nlp,
                    n_process=1,        # Use multiple CPU cores
                    batch_size=cfg.biber_tag.batch_size,     # Optimize batch size
                    show_progress=True  # Display progress bar
                )

                df_biber_tagged = pb.biber(df_tokens, force_ttr=True)

                # --- Assert correctness of Biber frequencies ---
                violating = df_biber_tagged.filter(pl.col("f_23_wh_clause") >= 1000)

                if violating.height > 0:
                    print("Bad rows in this batch:")
                    print(violating.select(["doc_id", "f_23_wh_clause"]))

                    bad_doc_id = violating["doc_id"][0]

                    tokens_bad = df_tokens.filter(pl.col("doc_id") == bad_doc_id)
                    print("token rows for bad_doc:", tokens_bad.height)
                    print(tokens_bad.select(["token", "tag", "pos", "dep_rel"]).head(50))
                    raise AssertionError("f_23_wh_clause exceeded 1000")

                # ------------------------------------------------
                
                with open(biber_output_path_ndjson, "a") as f:
                    f.write(df_biber_tagged.write_ndjson())

        gc.collect()
        print("Reading the ndjson result...")
        df_biber_tagged = pl.read_ndjson(biber_output_path_ndjson)
        print("Write to Parquet file...")

        sorted_cols = sorted(df_biber_tagged.columns)
        df_biber_tagged.write_parquet(os.path.join(experiment.path, f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet"))
        df_biber_tagged.select(sorted_cols).write_csv(os.path.join(experiment.path, f"dataset_{cfg.biber_tag.spacy_model}_biber.csv"))
        
        print("Removing the temporary ndjson file...")
        os.remove(biber_output_path_ndjson)
        print(f"Biber tagged DataFrame saved to {os.path.join(experiment.path, f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet")}")



@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")
    print(f"Dataset: {cfg.dataset.name}")
    biber_tag_experiments(cfg)

if __name__ == "__main__":
    main()



