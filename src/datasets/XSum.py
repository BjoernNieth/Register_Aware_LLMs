import os
import pandas as pd
import shutil
from datasets import load_dataset
from .Base_Dataset import Base_Dataset_Class

# Limit size of XSum dataset
#max_length=10000
class XSum(Base_Dataset_Class):
    """ XSum dataset from 2010 - 2017"""
    def __init__(self, data_path, cutoffdate, mfte_tagger_path, n_shot=0):
        super().__init__()
        if not os.path.isfile(os.path.join(data_path, "XSum.csv")):
            self.dataset = self._prepare_dataset(data_path, cutoffdate, mfte_tagger_path)
        else:
            self.dataset = pd.read_csv(os.path.join(data_path, "XSum.csv")).set_index("id")
        self.i = None
        self.n_shot = n_shot

    def __iter__(self):
        self.i = 0
        return self
    
    def __len__(self):
        return len(self.dataset.index)
    
    def __next__(self):
        """ Iterator for the dataset class, which will return the sample along with n randomly selected n-shot samples"""
        self.i += 1
        if self.i >= len(self.dataset.index):
            raise StopIteration

        if self.n_shot > 0:
            few_shot_samples = self.dataset.drop(self.dataset.index[self.i - 1]).sample(n=self.n_shot)
        else:
            few_shot_samples = None

        return self.dataset.index[self.i - 1], self.dataset.iloc[self.i - 1], few_shot_samples
    
    def set_n_shot(self, n_shot):
        self.n_shot = n_shot

    def _prepare_dataset(self, data_path, cutoffdate, mfte_tagger_path):
        """ Download the dataset, apply MFTE tagger and bring the data into the correct format """
        cache_dir = os.path.join(data_path, "temp_cache", "XSum")
        dataset = self._download_dataset(cache_dir)
        dataset = self._clean_dataset(dataset)

        # Directory where the MFTE tagger will store the output
        mfte_results_path = os.path.join(cache_dir, "XSum_Corpus_MFTE", "Statistics", "counts_word-based_normed.csv")
        if not os.path.isfile(mfte_results_path):
            self._mfte_tagging(dataset, cache_dir, mfte_tagger_path)
        # Read in the results of the MFTE tagger
        features = pd.read_csv(mfte_results_path, sep=",", header=0)

        # Convert filename back to index name
        features["Filename"] = features[["Filename"]].replace(".txt", "", regex=True)
        features = features.rename(columns={"Filename": "id"})

        
        print(features.columns)
        features = features.set_index("id")

        # Merge features and dataset
        dataset = dataset.join(features)

        dataset.to_csv(os.path.join(data_path, "XSum_Full.csv"))
        dataset = self._clean_dataset(dataset)

        dataset.to_csv(os.path.join(data_path, "XSum_Cleaned.csv"))

        #dataset = dataset.sample(n=max_length)
        dataset.to_csv(os.path.join(data_path, "XSum.csv"))

        # Remove the temporary files
        shutil.rmtree(os.path.join(cache_dir, "XSum_Corpus"))
        shutil.rmtree(os.path.join(cache_dir, "XSum_Corpus_MFTE", "POS_Tagged"))
        shutil.rmtree(os.path.join(cache_dir, "XSum_Corpus_MFTE", "MFTE_Tagged"))

        return dataset


    def _mfte_tagging(self, dataset, cache_dir, mfte_tagger_path):
        """ Write the dataset into a text corpus and apply the MFTE tagger to it """
        # Directory where the single txt files will be stored
        txt_path = os.path.join(cache_dir, "XSum_Corpus")


        print(f"Write the text Corpus in to temporary Cache at: {txt_path}")
        os.makedirs(txt_path, exist_ok=True)
        # Write each sample of the corpus into a txt file for the MFTE tagger
        for index, row in dataset.iterrows():
            file_path = os.path.join(txt_path,  str(index) + ".txt")
            if not os.path.exists(file_path):
                with open(file_path, "x", encoding='utf-8') as f:
                    f.write(row["document"].replace('\n', ' ').replace('  ', ' ').replace('\r', ''))

        print(f"Call the mfte tagger with: python {mfte_tagger_path} --path {txt_path} --parallel_md_tagging True > /dev/null 2>&1")
        # Call the MFTE tagger on the corpus
        os.system(f'python {mfte_tagger_path} --path {txt_path} --parallel_md_tagging True --extended False > /dev/null 2>&1')




    def _download_dataset(self, cache_dir):
        """Download the Train, Validation and Test dataset from XSum"""

        train_set = load_dataset("EdinburghNLP/xsum", cache_dir=cache_dir,
                                    trust_remote_code=True)["train"]
        validation_set = load_dataset("EdinburghNLP/xsum", cache_dir=cache_dir,
                                    trust_remote_code=True)["validation"]
        test_set = load_dataset("EdinburghNLP/xsum", cache_dir=cache_dir,
                                    trust_remote_code=True)["test"]

        dataset = pd.concat([train_set.to_pandas(), validation_set.to_pandas(), test_set.to_pandas()])
        dataset = dataset.set_index("id")
        return dataset


    def _clean_dataset(self, dataset):
        """ Remove all new articles smaller than 1000 characters and larger than 2000 characters"""
        # Add the length of the string columns as an extra column to the dataframe
        for col in dataset.select_dtypes(include=['object']):
            # Remove all multiple whitespaces and formatting signs
            dataset[col] = dataset[col].apply(lambda a: " ".join(a.split()))
            dataset[f'{col}_length'] = dataset[col].astype(str).apply(len)

        dataset = dataset[dataset["document_length"] < 2000]
        dataset = dataset[dataset["document_length"] > 1000]

        return dataset
