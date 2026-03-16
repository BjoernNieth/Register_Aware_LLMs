# How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework

Code and artefacts for the paper "How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework". This repository includes the source code to reproduce the results, as well as the precomputed results under `results/`.

## How to run

Create the Conda environment first:

```bash
conda env create -f environment.yml
conda activate vllm
```

The dataset preparation and post-processing of the generations are handled through scripts that can be called through the `Makefile` in the project root. The project uses Hydra to configure these scripts.

Set the dataset with `DATASET=...` and the location where datasets and results are stored with `DATADIR=...`.

Example:

```bash
make prepare_experiments DATASET=WritingPrompts DATADIR=/example/path
make stability_analysis DATASET=WritingPrompts DATADIR=/example/path
make get_biber_features_experiment DATASET=WritingPrompts DATADIR=/example/path
make get_biber_dimensions DATASET=WritingPrompts DATADIR=/example/path
make get_stat_evaluation DATASET=WritingPrompts DATADIR=/example/path
```

The available datasets in this repository are:

- `WritingPrompts`
- `XSum`
- `wikiHow`
- `S2ORC_ACL`
- `BNC2014Spoken`

## Run Generation

Generation setups are stored in `experiment_configs/`.

- `experiment_configs/`: main generation runs across datasets, models, and shot settings
- `experiment_configs/ablation/`: prompt ablation runs

Model generation is launched with:

```bash
python run_pipeline.py \
  --data-path <path-to-data> \
  --output-dir <path-to-output-dir> \
  --huggingface-token <hf-token> \
  --config-file-path <experiment-config.json>
```

Example:

```bash
python run_pipeline.py \
  --data-path /path/to/datasets+experiments/WritingPrompts/data \
  --output-dir /path/to/datasets+experiments/WritingPrompts/experiments \
  --huggingface-token <hf-token> \
  --config-file-path experiment_configs/WritingPrompts_Zero_Shot_Qwen_8B.json
```

Each JSON file contains one or more experiment definitions, including the dataset, model, prompt template, decoding parameters, shot setting, and seed.

The configs used for the experiments can be found in `experiment_configs` or within the respective experiment folder in `results`
