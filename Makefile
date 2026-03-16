DATADIR ?= .\results\datasets+experiments
DATASET ?= WritingPrompts

prepare_experiments: prepare_data get_biber_features subsample

prepare_data: 
	python scripts/01_prepare_data.py dataset=$(DATASET)

get_biber_features: $(DATADIR)/$(DATASET)/data/Dataset.csv
	python scripts/02_get_biber_features.py dataset=$(DATASET)

stability_analysis: $(DATADIR)/$(DATASET)/data/Dataset.csv
	python scripts/03_2_validate_sample_size.py dataset=$(DATASET)

subsample: $(DATADIR)/$(DATASET)/data/Dataset.csv
	python scripts/03_subsample_data.py dataset=$(DATASET)

get_biber_features_experiment: $(DATADIR)/$(DATASET)/experiments/
	python scripts/04_get_biber_features_experiments.py dataset=$(DATASET)

get_biber_dimensions: $(DATADIR)/$(DATASET)/experiments/ $(DATADIR)/$(DATASET)/data/Dataset.csv
	python scripts/05_OG_Biber.py dataset=$(DATASET)

get_stat_evaluation: $(DATADIR)/$(DATASET)/artefacts/Dimensional_Loadings_1988.csv
	python scripts/06_Stat_analysis.py dataset=$(DATASET)