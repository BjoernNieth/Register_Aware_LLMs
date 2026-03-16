import os
import pandas as pd
import shutil
from datasets import load_dataset
from .Base_Dataset import Base_Dataset_Class

# Limit the WritingPrompts dataset to 30000 samples
max_length=10000
class WritingPrompts(Base_Dataset_Class):
    def __init__(self, data_path, cutoffdate, mfte_tagger_path, n_shot=0):
        if not os.path.isfile(os.path.join(data_path, "WritingPrompts.csv")):
            self.dataset = self._prepare_dataset(data_path, cutoffdate, mfte_tagger_path)
        else:
            self.dataset = pd.read_csv(os.path.join(data_path, "WritingPrompts.csv")).set_index("id")
        self.i = None
        self.n_shot = n_shot

    def _prepare_dataset(self, data_path, cutoffdate, mfte_tagger_path):
        """ Download the dataset, apply MFTE tagger and bring the data into the correct format """
        cache_dir = os.path.join(data_path, "temp_cache", "WritingPrompts")
        dataset = self._download_dataset(cache_dir)
        dataset.to_csv(os.path.join(data_path, "WritingPrompts_Full.csv"))
        dataset = self._clean_dataset(dataset)
        dataset.to_csv(os.path.join(data_path, "WritingPrompts_Clean.csv"))
        dataset = dataset.sample(n=max_length)

        # Directory where the MFTE tagger will store the output
        mfte_results_path = os.path.join(cache_dir, "WritingPrompts_Corpus_MFTE", "Statistics", "counts_word-based_normed.csv")
        if not os.path.isfile(mfte_results_path):
            self._mfte_tagging(dataset, cache_dir, mfte_tagger_path)
        # Read in the results of the MFTE tagger
        features = pd.read_csv(mfte_results_path, sep=",", header=0)

        # Convert filename back to index name
        features["Filename"] = features[["Filename"]].replace(".txt", "", regex=True)
        features = features.rename(columns={"Filename": "id"})
        features["id"] = features["id"].astype(int)
        features = features.set_index("id")

        # Merge features and dataset
        dataset = dataset.join(features)
        # Save the stored
        dataset.to_csv(os.path.join(data_path, "WritingPrompts.csv"))

        # Remove temporary files
        shutil.rmtree(os.path.join(cache_dir, "WritingPrompts_Corpus"))
        shutil.rmtree(os.path.join(cache_dir, "WritingPrompts_Corpus_MFTE", "POS_Tagged"))
        shutil.rmtree(os.path.join(cache_dir, "WritingPrompts_Corpus_MFTE", "MFTE_Tagged"))

        return dataset

    def tag_results(self, results_df, cache_dir, output_path, mfte_tagger_path):
        output_folder_name = os.path.basename(os.path.normpath(output_path))

        cache_corpus_dir = os.path.join(cache_dir, "temp_cache", output_folder_name)
        self._mfte_tagging(results_df, cache_corpus_dir, mfte_tagger_path)
        mfte_results_path = os.path.join(cache_corpus_dir, "WritingPrompts_Corpus_MFTE", "Statistics", "counts_word-based_normed.csv")

        features = pd.read_csv(mfte_results_path, sep=",", header=0)
        # Convert filename back to index name
        features["Filename"] = features[["Filename"]].replace(".txt", "", regex=True)
        features = features.rename(columns={"Filename": "id"})
        features = features.set_index("id")
        results_df = results_df.join(features)
        result_path = os.path.join(output_path, "WritingPrompts.csv")

        print(f"Print XSum results to {result_path}")
        results_df.to_csv(result_path)

        return


    def _mfte_tagging(self, dataset, cache_dir, mfte_tagger_path):
        """ Write the dataset into a text corpus and apply the MFTE tagger to it """
        # Directory where the single txt files will be stored
        txt_path = os.path.join(cache_dir, "WritingPrompts_Corpus")
        print(f"Write the text Corpus in to temporary Cache at: {txt_path}")
        os.makedirs(txt_path, exist_ok=True)
        # Write each sample of the corpus into a txt file for the MFTE tagger
        for index, row in dataset.iterrows():
            file_path = os.path.join(txt_path,  str(index) + ".txt")
            if not os.path.exists(file_path):
                with open(file_path, "x", encoding='utf-8') as f:
                    f.write(row["story"].replace('\n', ' ').replace('  ', ' ').replace('\r', ''))

        print(f"Call the mfte tagger with: python {mfte_tagger_path} --path {txt_path} --parallel_md_tagging True > /dev/null 2>&1")
        # Call the MFTE tagger on the corpus
        os.system(f'python {mfte_tagger_path} --path {txt_path} --parallel_md_tagging True > /dev/null 2>&1')


    def _download_dataset(self, cache_dir):
        """Download the WritingPrompts dataset"""

        train_set = load_dataset("euclaise/writingprompts", cache_dir=cache_dir,
                                    trust_remote_code=True)["train"]
        validation_set = load_dataset("euclaise/writingprompts", cache_dir=cache_dir,
                                    trust_remote_code=True)["validation"]
        test_set = load_dataset("euclaise/writingprompts", cache_dir=cache_dir,
                                    trust_remote_code=True)["test"]

        # Merge the splitted dataset and reindex it
        dataset = pd.concat([train_set.to_pandas(), validation_set.to_pandas(), test_set.to_pandas()], ignore_index=True)
        # Explicitly name the index
        dataset.index.name = "id"

        return dataset

    def _clean_dataset(self, dataset):
        # Only keep general writing prompts indicated by [ WP ]
        dataset = dataset[dataset["prompt"].apply(lambda a: "[ WP ]" in a)]
        # Remove starting tag from prompt
        dataset["prompt"] = dataset["prompt"].str[6:]
        # Add the length of the string columns as an extra column to the dataframe
        for col in dataset.select_dtypes(include=['object']):
            # Remove multiple whitespaces and formatting keys
            dataset[col] = dataset[col].apply(lambda a: " ".join(a.split()))
            dataset[f'{col}_length'] = dataset[col].astype(str).apply(len)

        # Remove all prompts shorter than roughly 5 words
        dataset = dataset[dataset["prompt_length"] > 6 * 5]

        # Avoid generating answers longer than 5000 characters
        dataset = dataset[dataset["story_length"] < 5000]

        return dataset