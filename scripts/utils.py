import re
import unicodedata
import os
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import re
import spacy
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from faker import Faker
import numpy as np
from scipy.stats import wasserstein_distance


nlp = spacy.load("en_core_web_sm", disable=["ner"])

# Read in the biber loadings of the original 1988 Biber study. 
# Drop f_62_split_infinitive as it is not used in bibers analysis and the mean is zero
biber_loadings_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "biber_loadings", "biber_loadings_1988.csv")).drop(["f_62_split_infinitive"], axis=1)
biber_normalizations_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "biber_loadings", "biber_normalisations_1988.csv"))
biber_normalizations_df = biber_normalizations_df[~biber_normalizations_df["Feature"].isin(["f_62_split_infinitive"])]

def _get_tokenized_length(x, tokenizer, columns):
    token_length = 0
    for col in columns:
        token_length += len(tokenizer.encode(x[col]))

    return token_length

def get_tokenized_length(df, columns, hf_token, model="meta-llama/Llama-3.3-70B-Instruct"):
    from transformers import AutoTokenizer
    from huggingface_hub import login
    login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    return df.apply(_get_tokenized_length, args=(tokenizer, columns), axis=1)

def coerce_numeric(df):
    # Ensure numeric float64 everywhere
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out

def get_dimensional_loading(biber_features):
    if "f_62_split_infinitive" in biber_features.columns:
        biber_features = biber_features.drop("f_62_split_infinitive", axis=1)
    biber_features = coerce_numeric(biber_features)
    f_size = biber_normalizations_df.values.shape[0]

    # Normalize the features with the mean and std from the original Biber 1988 study
    normalized_features = (biber_features.values - biber_normalizations_df["Mean"].values.reshape((1,f_size))) / biber_normalizations_df["Std"].values.reshape((1,f_size))

    results = { "doc_id": list(biber_features.index) }
    # Calculated the dimensional loadings for the 1988 study
    for i in range(biber_loadings_df.values.shape[0]):
        results[f"dimension_{i + 1}"] = (normalized_features * biber_loadings_df.drop("dimension",axis=1).values[i]).sum(axis=1).tolist()

    return pd.DataFrame(results).set_index("doc_id")

def dimensional_matching_wasserstein(biber_df_full, biber_df_subsampled):
    full_dimensions = biber_df_full.values
    subsampled_dimension = biber_df_subsampled.values

    
    dists = []
    for d in range(subsampled_dimension.shape[1]):
        Wd = wasserstein_distance(full_dimensions[:, d], subsampled_dimension[:, d])
        dists.append(Wd)
    return np.mean(dists)

def string_cleaning(x):
    """ Perform standard String cleaning operations"""
    # Remove formating
    x = re.sub(r'[\r\n\t]+', ' ', x)
    # Remove double whitespaces
    x = re.sub(r'\s+', ' ', x).strip()
    # Remove spaces before punctuations 
    x = re.sub(r"\s+([.,?!'])", r'\1', x)
    # Ensure same encoding is used
    return unicodedata.normalize("NFKC", x)

def punctuation_ratio(text):
    punct_count = len(re.findall(r'[^\w\s]', text))
    return punct_count / (len(text) + 1)

def ensure_exists_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def get_parsed_text(texts: list):
    # Parse the texts into spacy
    docs = list(tqdm(nlp.pipe(texts, n_process=-1, batch_size=100), total=len(texts)))
    return docs

def word_count_and_shortend(texts, max_words = 400, cap = 440, n_process=4):
    nlp_clean = spacy.load("en_core_web_sm", disable=["ner"])
    word_counts = [None] * len(texts)
    capped_texts = [None] * len(texts)

    for i, doc in enumerate(tqdm(nlp_clean.pipe(texts, n_process=n_process, batch_size=64), total=len(texts))):
        word_counts[i] = count_words_doc(doc)
        capped_texts[i] = window_text_soft_doc(doc, max_words, cap)

    return word_counts, capped_texts    

def count_words_doc(doc):
    return sum(1 for t in doc if not (t.is_punct or t.is_space))

def count_words(text) -> int:
    """Quick orthographic word counter (PTB tokenization; keeps clitics, drops pure punctuation)."""
    doc = nlp(text)
    return count_words_doc(doc)

def _sent_word_count(sent) -> int:
    return sum(1 for t in sent if not (t.is_punct or t.is_space))

def window_text_soft_doc(doc, max_words = 400, cap = 440) -> str:
    selected = []
    total_words = 0

    # Helper to count words inside a sentence (no punctuation/space)
    for sent in doc.sents:
        swc = _sent_word_count(sent)

        if total_words < max_words + cap:
            # Only add sentence if it is still within soft cap
            selected.append(sent)
            total_words += swc

            if total_words > max_words:
                break
        else:
            break 


    # Enfoce the hard-cap even if mid-sentence. This might happen due to LLM output
    parts = []
    word_count = 0
    for sent in selected:
        for token in sent:

            # Add the token to the output. 
            parts.append(token.text_with_ws)

            # Count only word tokens
            if not (token.is_punct or token.is_space):
                word_count += 1
                if word_count >= cap:
                    # Return everything up to this point.
                    return "".join(parts).strip()

    return "".join(parts).strip()

def window_text_soft(text, max_words = 400, cap = 440) -> str:
    """
    Soft window: keep up to `max_words` *wordlike* tokens, then extend to next sentence boundary,
    but never exceed `cap` total tokens (word + punctuation). Uses PTB tokenization and detokenizes.
    """
    doc = nlp(text)
    return window_text_soft_doc(doc, max_words, cap)

def plot_length_hist(df, text_column, output_path, bins=50):
    """
    Compute word counts for a column of texts and plot a histogram.

    Args:
        df (pd.DataFrame): DataFrame with a text column.
        text_column (str): name of the column containing text.
        bins (int): number of histogram bins (default 50).
    """
    lengths = df[text_column].apply(count_words)

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=bins, edgecolor="black")
    plt.xlabel("Word length")
    plt.ylabel("Number of texts")
    plt.title(f"Histogram of text lengths in '{text_column}'")
    plt.savefig(output_path)




def clean_names(names):
    cleaned = []
    for name in names:
        # 1. Remove anything in parentheses, like "(name)" or "(given name)"
        name = re.sub(r'\s*\(.*?\)\s*$', '', name).strip()
        
        # 2. Skip names with non-ASCII letters (like Æ, ñ, ö, etc.)
        if any(ord(c) > 127 for c in name):
            continue
        
        # 3. Optionally: only keep names with regular English characters, hyphens, or apostrophes
        if not re.match(r"^[A-Za-z][A-Za-z'\- ]*[A-Za-z]$|^[A-Za-z]$", name):
            continue
        
        cleaned.append(name)
    
    return cleaned

def get_names(sex):
    base = "https://en.wikipedia.org"
    if sex == "female":
        url = f"{base}/wiki/Category:English_feminine_given_names"
    elif sex == "male":
        url = f"{base}/wiki/Category:English_masculine_given_names"
    headers = {"User-Agent": "Mozilla/5.0"}
    #https://en.wikipedia.org/wiki/Category:English_masculine_given_names
    all_names = []
    
    while url:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        all_names += [li.text.strip() for li in soup.select("div#mw-pages li")]
    
        next_link = soup.select_one("a:contains('next page')")
        url = urljoin(base, next_link["href"]) if next_link else None
        
    return clean_names(all_names)

def capitalize_first(text):
    return text[0].upper() + text[1:] if text else text

# --- Examples ---
if __name__ == "__main__":
    s = "The pain hadn 't become unbearable... He found his existence pointless in every aspect. Having lost the will to enjoy life many years ago, at this point he was only going through the motions. He felt as though he lived life on autopilot; giving new meaning to the old saying “ the lights were on but no one was home ”. Years of self-medicating through pills, alcohol, and a razor he used for far more than just shaving his beard, he decided enough was enough. It had been 25 years, and nothing has changed, and he didn ’ t foresee an option in his future where it did. It was after this epiphany when he finally came to the conclusion that he just didn ’ t want to be here anymore. That life wasn ’ t, in fact going to get any better, and he was so tired of random internet strangers telling him to hold on because there was a “ chance ”. Nothing really mattered anymore, and even if it did, he wanted no part of it. 2am on the quietest Monday night this bridge has ever seen, he parked his car on the shoulder and sat with his passenger, Jack Daniels while he attempted to collect his racing thoughts. Growing more agitated by the second, he realized the loudness in his head would never cease, and even the whiskey wouldn ’ t work anymore. In a panic he jumped out of his car, and leaned himself over the railing, his head spinning. After hoisting himself over onto the other side, he closed his eyes and promised he wouldn ’ t give himself the time to overthink it. At least he would be at peace, he thought as he let himself fall. He opened his eyes as the sound of angry dark water furiously churning below him became louder and louder."
    print("len:", count_words(s))                 # e.g., 10 (punctuation excluded)
    print("simple len", len(s.split()))
    print(window_text_soft(s, 4, 3))           # keep full sentences up to n..n+m
    print(window_text_soft(s, 8, 8))