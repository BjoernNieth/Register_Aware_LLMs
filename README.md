# How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework

This repository contains the code and the results for the EMNLP 2026 publication: "How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework"

We compare human and LLM-generated text using the commonly applied linguistic framework of Biber (1988). In the paper we analyse five different datasets spanning five different registers: WritingPrompts (fiction), XSum (news), wikiHow (instructional), S2ORC_ACL (academic writing) and BNC2014Spoken (conversation). On these datasets we test seven open-weight models
(Apertus-70B, Llama-3.3-70B, Llama-3.1-8B, Qwen3-32B, Qwen3-8B, Gemma-3-27B, Gemma-3-12B) in a zero- to three-shot setting. Additionally we run promp ablation experiments for all models on the BNC2014Spoken dataset.

## Setup

```bash
conda env create -f environment.yml
conda activate vllm
```

## Getting the data

Due to licencing constraints not all datasets are published with this repo. Running the code WritingPrompts and XSum are automatically downloaded from Hugging Face. To use the other datasets you first need to do the following:

- **S2ORC_ACL** — put a Semantic Scholar API key in `configs/dataset/S2ORC_ACL.yaml`. The
  release is pinned to `2022-05-10` and filtered to ACL long papers from 2009–2018. 
- **wikiHow** — download `wikihowAll.csv.zip` (Koupaee & Wang, 2018) and place it in `DATADIR`. For the paper we used the [version published on Kaggle](https://www.kaggle.com/datasets/varunucl/wikihow-summarization).
- **BNC2014Spoken** — the Spoken BNC2014 requires registering with Lancaster University and
  accepting a licence. Place the XML release at `$DATADIR/bnc2014spoken-xml.zip`. [http://corpora.lancs.ac.uk/bnc2014/](http://corpora.lancs.ac.uk/bnc2014/)

Because of the licenses, no BNC2014Spoken or XSum source text is included in this repository. However, we publish the Biber features we used to compute our results of the paper.

## Reproducing the results

The analysis runs off the Biber features shipped in `results/` using the following commmand:

```bash
for DS in WritingPrompts XSum wikiHow S2ORC_ACL BNC2014Spoken; do
  make reproduce DATASET=$DS
done
```

This writes into `results/datasets+experiments/<DATASET>/artefacts/` the dimension loadings, MMD and
Wasserstein results, stability curves and variances. Figures come from the notebook:

```bash
cd visualisations
PYTHONPATH=../scripts jupyter notebook Visualize_Results.ipynb
```

The `PYTHONPATH` is needed because the notebook imports `stats.py` from `scripts/`.

## Running the full pipeline

Preparing a corpus from scratch and re-running generation needs the data above and a GPU. The
70B configs assume `tensor_parallel_size: 2`. In our experiments we used, depending on the model, one or two A100 with 80GB of vRAM

```bash
# This prepares the raw dataset into the subsampled human dataset
make prepare_experiments DATASET=WritingPrompts DATADIR=/path/to/data   # 01, 02, 03

# Run the LLM generations
python run_pipeline.py \
  --data-path  /path/to/data/WritingPrompts/data \
  --output-dir /path/to/data/WritingPrompts/experiments \
  --huggingface-token <hf-token> \
  --config-file-path experiment_configs/WritingPrompts_Zero_Shot_Qwen_8B.json

# Get the Biber features of the generated texts
make get_biber_features_experiment DATASET=WritingPrompts                # 04

# Get the result dataframes
make reproduce DATASET=WritingPrompts                                    # 05, 06
```

Stages in order:

| | | 
|---|---|
| `01_prepare_data.py` | corpus → `data/Dataset.csv` | 
| `02_get_biber_features.py` | Biber-tag the human texts |
| `03_subsample_data.py` | draw the 600-doc sample and few-shot pool |
| `03_2_validate_sample_size.py` | sample-size stability check (optional) |
| `run_pipeline.py` | generate model outputs |
| `04_get_biber_features_experiments.py` | Biber-tag the generations |
| `05_OG_Biber.py` | dimension loadings |
| `06_Stat_analysis.py` | MMD, Wasserstein, stability, variances |

Subsampling is seeded (`seed: 127` in `configs/run/default.yaml`), so the splits reproduce
exactly given the same corpus.

## Configuration

For reproducability, we set up all experiments using Hydra:

```bash
python scripts/06_Stat_analysis.py dataset=XSum datapath=/path/to/data
```

`DATASET` and `DATADIR` in the Makefile map to `dataset=` and `datapath=`. Two placeholders
need filling before the relevant stage runs: `hf_token` in `configs/run/default.yaml` (used by
the tokenizer during subsampling) and `api_key` in `configs/dataset/S2ORC_ACL.yaml`.

## Layout

```
configs/              Hydra config groups
experiment_configs/   generation configs
scripts/              numbered pipeline stages, plus the Biber 1988 loadings
src/                  generation code (datasets, model wrapper, prompters)
visualisations/       Visualize_Results.ipynb
results/datasets+experiments/<DATASET>/
    data/             human Biber features and the split definitions
    experiments/      one directory per run: generations, features, config
    artefacts/        statistics, regenerated by `make reproduce`
```

## Citation

```bibtex
@inproceedings{nieth-etal-2026-human,
    title = "How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework",
    author = "Nieth, Bj{\"o}rn  and
      Gracheva, Marianna  and
      Mahlberg, Michaela  and
      Eskofier, Bjoern  and
      Salin, Emmanuelle",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
}
```

