
DATADIR ?= ./results/datasets+experiments
DATASET ?= WritingPrompts

HYDRA = dataset=$(DATASET) datapath=$(DATADIR)

.PHONY: prepare_experiments prepare_data get_biber_features stability_analysis \
        subsample get_biber_features_experiment get_biber_dimensions \
        get_stat_evaluation reproduce

# --- Stage 1: corpus preparation (needs the corpus; 02 needs a GPU) ----------
prepare_experiments: prepare_data get_biber_features subsample

prepare_data:
	python scripts/01_prepare_data.py $(HYDRA)

get_biber_features:
	python scripts/02_get_biber_features.py $(HYDRA)

subsample:
	python scripts/03_subsample_data.py $(HYDRA)

stability_analysis:
	python scripts/03_2_validate_sample_size.py $(HYDRA)


get_biber_features_experiment:
	python scripts/04_get_biber_features_experiments.py $(HYDRA)

get_biber_dimensions:
	python scripts/05_OG_Biber.py $(HYDRA)

get_stat_evaluation:
	python scripts/06_Stat_analysis.py $(HYDRA)

reproduce: get_biber_dimensions get_stat_evaluation
