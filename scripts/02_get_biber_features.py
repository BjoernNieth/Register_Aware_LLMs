import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
import pandas as pd 
import polars as pl
import pybiber as pb
import spacy
import gc
import tqdm
import time
import inspect
print(os.path.dirname(inspect.getfile(pb)))
def biber_tag_dataset(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    spacy.require_gpu()

    print("Setting up the Biber tagging script...")

    print(f"Processing CSV file: {os.path.join(PREPARED_DATA_PATH, "Dataset.csv")}")
    # Read DF into Polars
    dataset = pl.read_csv(os.path.join(PREPARED_DATA_PATH, "Dataset.csv"), has_header=True)
    dataset = dataset.rename({dataset.columns[0]: "doc_id"})

    print("Transforming the dataset into the correct format for PyBiber...")
    # Create a df in the correct format for pybiber
    biber_df = dataset.select(
        pl.col("doc_id").cast(pl.String),
        pl.col(cfg.dataset.text_name).alias("text")
    )
    del dataset
    gc.collect()
    
    print("Loading spaCy model...")
    nlp = spacy.load(cfg.biber_tag.spacy_model, disable=["ner"])
    
    print("Sorting the DataFrame by text length to optimize processing...")
    assert biber_df.filter(pl.col("text").is_not_null()).height == biber_df.height

    # Sort after text length to speed up inference by avoding padding short texts to longest text in batch
    # Add text_length and sort, then drop text_length
    biber_df = biber_df.with_columns(
        pl.col("text").str.len_chars().alias("text_length")
    ).sort("text_length", descending=True)
    biber_df = biber_df.drop("text_length")

    print("Check for already processed samples...")
    biber_output_path_ndjson = os.path.join(CACHE_PATH,  cfg.dataset.name + f"_{cfg.biber_tag.spacy_model}_biber.ndjson")
    if os.path.isfile(biber_output_path_ndjson):
        initital_len = biber_df.height
        # Remove all the already tagged texts from the df
        biber_df = biber_df.filter(~biber_df["doc_id"].is_in(pl.read_ndjson(biber_output_path_ndjson)["doc_id"]))
        print(f"{initital_len - biber_df.height} Samples were already Biber tagged!")

    
    if  biber_df.height > 0:
        print("Starting spaCy parsing and Biber Tagging...")
        for i in tqdm.tqdm(range(0, biber_df.height, cfg.biber_tag.batch_size)):
            # Reinnit processor, as this solved the bug with f_23 begin very high
            processor = pb.CorpusProcessor()
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


    print("Finished spaCy parsing and Biber Tagging.")
    # Clean up memory 
    del biber_df
    #del df_biber_tagged
    del nlp

    gc.collect()
    print("Reading the ndjson result...")
    df_biber_tagged = pl.read_ndjson(biber_output_path_ndjson)
    print("Write to Parquet file...")
    df_biber_tagged.write_parquet(os.path.join(DATASET_PATH, "data", f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet"))
    
    print("Removing the temporary ndjson file...")
    os.remove(biber_output_path_ndjson)
    print("Done!")
    print(f"Biber tagged DataFrame saved to {os.path.join(DATASET_PATH, "data", f"dataset_{cfg.biber_tag.spacy_model}_biber.parquet")}")



@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")

    print(f"Dataset: {cfg.dataset.name}")

    biber_tag_dataset(cfg)

if __name__ == "__main__":
    main()



