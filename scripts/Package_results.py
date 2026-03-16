import re

import os

import argparse
import pandas as pd 
from pathlib import Path

def parse_group_fields(group_value):
    if group_value == "Human":
        return {
            "Authorship": "Human",
            "Model": "Human",
            "Model_Size": "Human",
            "Strategy": -1,
            "Ablation": -1,
        }
    model_part = re.split(r'_(Zero|\d+)_Shot_', group_value, maxsplit=1)[-1]
    model_name, model_size = re.sub(r'_prompt_ablation_\d+$', '', model_part).split('_')
    m = re.search(r'_(Zero|\d+)_Shot_', group_value)
    num_shots = 0 if m.group(1) == "Zero" else int(m.group(1))
    m = re.search(r'_prompt_ablation_(\d+)', group_value)
    prompt_ablation = int(m.group(1)) if m else 0
    return {
        "Authorship": "AI",
        "Model": model_name,
        "Model_Size": model_size,
        "Strategy": num_shots,
        "Ablation": prompt_ablation,
    }

def package_all(study_path):
    biber_features_dfs = []
    texts_dfs = []
    dimensional_dfs = []
    for dataset, text_column in zip(["BNC2014Spoken", "S2ORC_ACL", "wikiHow", "WritingPrompts", "XSum"], ["Conversation_Clean", "introduction", "text", "story", "document"]):
        for experiment in os.scandir(os.path.join(study_path, dataset, "experiments")):
            if experiment.is_dir():
                biber_features_df = pd.read_parquet(os.path.join(experiment.path, f"dataset_en_core_web_trf_biber.parquet"))
                biber_features_df["Authorship"] = "AI"
                biber_features_df["Register"] = dataset

                model_part = re.split(r'_(Zero|\d+)_Shot_', experiment.name, maxsplit=1)[-1]
                model_name, model_size = re.sub(r'_prompt_ablation_\d+$', '', model_part).split('_')
                biber_features_df["Model"] = model_name
                biber_features_df["Model_Size"] = model_size

                m = re.search(r'_(Zero|\d+)_Shot_', experiment.name)
                num_shots = 0 if m.group(1) == "Zero" else int(m.group(1))
                biber_features_df["Strategy"] =  num_shots

                m = re.search(r'_prompt_ablation_(\d+)', experiment.name)
                prompt_ablation = int(m.group(1)) if m else 0
                biber_features_df["Ablation"] = prompt_ablation
                
                text_df = pd.read_csv(os.path.join(experiment.path, "Model_Output_400_words.csv"), index_col=0)
                text_df = text_df[["generated_text"]].rename(columns={"generated_text": "Text"})
                text_df = text_df.reset_index(names="doc_id")
                text_df["Authorship"] = "AI"
                text_df["Register"] = dataset 
                text_df["Model"] = model_name
                text_df["Model_Size"] = model_size
                text_df["Strategy"] =  num_shots
                text_df["Ablation"] = prompt_ablation

                biber_features_dfs.append(biber_features_df)
                texts_dfs.append(text_df)

        dimensional_df = pd.read_csv(os.path.join(study_path, dataset, "artefacts", "Dimensional_Loadings_1988.csv"), index_col=0)
        dimensional_df = dimensional_df[~dimensional_df["group"].isin(["Human_Full", "Human_Few_Shot"])]
        dimensional_df["Register"] = dimensional_df["dataset"]
        parsed_fields = dimensional_df["group"].apply(parse_group_fields).apply(pd.Series)
        dimensional_df = pd.concat([dimensional_df, parsed_fields], axis=1)
        dimensional_df = dimensional_df.drop(columns=["group", "dataset"])
        dimensional_dfs.append(dimensional_df)

        biber_features_df = pd.read_csv(os.path.join(study_path, dataset, "data", "Dataset_Subsampled_Biber_Features.csv"), index_col=0)
        biber_features_df["Authorship"] = "Human"
        biber_features_df["Register"] = dataset
        biber_features_df["Model"] = "Human"
        biber_features_df["Strategy"] = -1
        biber_features_df["Ablation"] = -1
        biber_features_dfs.append(biber_features_df)

        text_df = pd.read_csv(os.path.join(study_path, dataset, "data", "Dataset_Subsampled.csv"), index_col=0)
        text_df = text_df[[text_column]].rename(columns={text_column: "Text"})
        text_df = text_df.reset_index(names="doc_id")
        text_df["Authorship"] = "Human"
        text_df["Register"] = dataset
        text_df["Model"] = "Human"
        text_df["Strategy"] = -1
        text_df["Ablation"] = -1
        texts_dfs.append(text_df)


    all_biber_features_df = pd.concat(biber_features_dfs, ignore_index=True)
    non_f = [c for c in all_biber_features_df.columns if not c.startswith("f_")]
    f_cols = sorted([c for c in all_biber_features_df.columns if c.startswith("f_")])
    all_biber_features_df = all_biber_features_df[non_f + f_cols]

    os.makedirs(os.path.join(study_path, "packaged_results"), exist_ok=True)
    all_biber_features_df.to_csv(os.path.join(study_path, "packaged_results", "all_biber_features.csv"), index=False)
    print(all_biber_features_df.head())

    all_dimensional_df = pd.concat(dimensional_dfs, ignore_index=True)
    all_dimensional_df.to_csv(os.path.join(study_path, "packaged_results", "all_dimensional_loadings.csv"), index=False)
    print(all_dimensional_df.head())

    all_texts_df = pd.concat(texts_dfs, ignore_index=True)
    all_texts_df.to_csv(os.path.join(study_path, "packaged_results", "all_texts.csv"), index=False)
    print(all_texts_df.head())

    

   

    print(dimensional_df.head())
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to a file or directory")
    args = parser.parse_args()

    study_path = args.path
    package_all(study_path)


