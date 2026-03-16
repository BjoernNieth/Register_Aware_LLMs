import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
from stats import estimate_median_heuristic, human_human_mmd_baseline, wasserstein_bootstrap, compute_mmd2_biased_batched, mmd_bootstrap, permutation_test_wasserstein_full, human_human_wasserstein_baseline, human_human_variance_baseline
from sklearn.preprocessing import StandardScaler
from utils import string_cleaning, coerce_numeric, punctuation_ratio, ensure_exists_dirs, plot_length_hist, count_words, window_text_soft
from scipy.stats import wasserstein_distance

import pandas as pd 
import polars as pl
import pybiber as pb
import spacy
import gc
import tqdm
import numpy as np
import time



def get_dimensional_loading(biber_features, biber_loadings, biber_normalizations):
    biber_features = coerce_numeric(biber_features)
    f_size = biber_normalizations.values.shape[0]

    # Normalize the features with the mean and std from the original Biber 1988 study
    normalized_features = (biber_features.values - biber_normalizations["Mean"].values.reshape((1,f_size))) / biber_normalizations["Std"].values.reshape((1,f_size))

    results = { "doc_ids": list(biber_features.index) }
    # Calculated the dimensional loadings for the 1988 study
    for i in range(biber_loadings.values.shape[0]):
        results[f"dimension_{i + 1}"] = (normalized_features * biber_loadings.drop("dimension",axis=1).values[i]).sum(axis=1).tolist()

    return pd.DataFrame(results)


def get_statistical_analysis(cfg):
    CACHE_PATH = os.path.join(cfg.datapath, ".cache")
    DATASET_PATH = os.path.join(cfg.datapath, cfg.dataset.name)
    EXPERIMENT_PATH = os.path.join(DATASET_PATH, "experiments")
    ARTEFACT_PATH = os.path.join(DATASET_PATH, "artefacts")
    PREPARED_DATA_PATH = os.path.join(DATASET_PATH, "data")
    
    stability_results = []
    mmd_results = []
    mmd_ablation_spread = []
    feature_results = []
    variances = []

    models = []
    biber_features_dfs = []
    print(EXPERIMENT_PATH)
    # Get a df with the biber features of all experiments
    for experiment in os.scandir(EXPERIMENT_PATH):
        if experiment.is_dir():
            try:
                df = pd.read_csv(os.path.join(experiment.path, f"dataset_en_core_web_trf_biber.csv"), index_col=0).sort_index()
                # Sort by column names
                df = df.reindex(sorted(df.columns), axis=1)

                df["Model"] = experiment.name
                models.append(experiment.name)
                biber_features_dfs.append(df)
                print(f"Loaded {experiment.name}")
            except:
                print(f"Skipping {experiment}")



    print("Load human data")
    df = pd.read_csv(os.path.join(PREPARED_DATA_PATH, "Dataset_Subsampled_Biber_Features.csv"), index_col=0).sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    feature_names = list(df.columns)
    df["Model"] = "Human"
    biber_features_dfs.append(df)
    biber_features_dfs = pd.concat(biber_features_dfs)
    print(f"Feature names: {feature_names}")

    print("Scale to full human dataset")
    # Fit a standard scaler to the full human biber feature distribution
    biber_features_df_human = pd.read_parquet(os.path.join(PREPARED_DATA_PATH, "dataset_en_core_web_trf_biber.parquet")).set_index("doc_id").sort_index()
    biber_features_df_human = biber_features_df_human.reindex(sorted(biber_features_df_human.columns), axis=1)
    feature_scaler = StandardScaler()
    feature_scaler.fit(biber_features_df_human.values)
    human_full_X_scaled = feature_scaler.transform(biber_features_df_human.values)

    print("Estimate median bandwith")
    # Estimate the bandwith for the MMD from the Human dataset
    sigma = estimate_median_heuristic(human_full_X_scaled)

    print("Get stability of MMD for sample size")
    # Via resampling get the CI for the HH-MMD-distance
    for n in tqdm.tqdm(range(cfg.subsample.sample_size_lower, cfg.subsample.sample_size_upper + 1, cfg.subsample.sample_size_step_size)):
        _, hh_mean, hh_std, ci_lower, ci_upper = human_human_mmd_baseline(human_full_X_scaled, sigma=sigma, n=n, R=cfg.stat_eval.resamples, ci=cfg.stat_eval.CI, seed=cfg.run.seed, centering=False)
        stability_results.append({
            "n": n,
            "mean_score": hh_mean,
            "std_score": hh_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "dataset": cfg.dataset.name,
            "source": "Human_Full",
            "type": "Human-Human",
            "method": "Resample", 
            "CI": cfg.stat_eval.CI,
            "sigma": sigma,
            "Normalized": "Human_Full"
        })
        _, hh_mean, hh_std, ci_lower, ci_upper = human_human_mmd_baseline(human_full_X_scaled, sigma=sigma, n=n, R=cfg.stat_eval.resamples, ci=cfg.stat_eval.CI, seed=cfg.run.seed, centering=True)
        stability_results.append({
            "n": n,
            "mean_score": hh_mean,
            "std_score": hh_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "dataset": cfg.dataset.name,
            "source": "Human_Full",
            "type": "Human-Human",
            "method": "Resample", 
            "CI": cfg.stat_eval.CI,
            "sigma": sigma,
            "Normalized": "Human_Full + Centering"
        })

        _, hh_mean, hh_std, lower, upper = human_human_variance_baseline(human_full_X_scaled, n=n, R=cfg.stat_eval.resamples, ci=cfg.stat_eval.CI, seed=cfg.run.seed)

        T = np.var(human_full_X_scaled, axis=0, ddof=1).sum()
        variances.append({
            "n": n,
            "variance": hh_mean,
            "variance_std": hh_std,
            "ci_lower": lower,
            "ci_upper": upper,
            "model": "Human_Full",
            "dataset": cfg.dataset.name
        })


    print("Calculate distance between human and model")
    human_X_scaled = feature_scaler.transform(biber_features_dfs[biber_features_dfs["Model"] == "Human"][feature_names].values)
    for model in models:
        print(f"Model: {model}")
        model_X_scaled = feature_scaler.transform(biber_features_dfs[biber_features_dfs["Model"] == model][feature_names].values)
        hm_observed = compute_mmd2_biased_batched(human_X_scaled, model_X_scaled, sigma=sigma)[0].cpu().item() 
        # Todo: MMD permutation test
        _, hm_mean, hm_std, lower, upper = mmd_bootstrap(human_X_scaled, model_X_scaled, sigma=sigma, R=cfg.stat_eval.resamples, ci=cfg.stat_eval.CI, seed=cfg.run.seed)
        mmd_results.append({
            "observed": hm_observed,
            "mean_score": hm_mean,
            "std_score": hm_std,
            "ci_lower": lower,
            "ci_upper": upper,
            "dataset": cfg.dataset.name,
            "model": model,
            "method": "Bootstrap", 
            "type": "Human-Model",
            "CI": cfg.stat_eval.CI,
            "sigma": sigma.cpu().item() 
        })

        hm_observed_zero_mean = compute_mmd2_biased_batched(human_X_scaled - human_X_scaled.mean(axis=0), model_X_scaled - model_X_scaled.mean(axis=0), sigma=sigma)[0].cpu().item()
        mmd_ablation_spread.append({
            "observed": hm_observed_zero_mean,
            "model": model,
            "type": "Human-Model_Zero_Mean"
        })

        model_variance = np.var(model_X_scaled, axis=0, ddof=1).sum()
        variances.append({
            "n": model_X_scaled.shape[0],
            "variance": model_variance,
            "variance_std": 0,
            "ci_lower": None,
            "ci_upper": None,
            "dataset": cfg.dataset.name,
            "model": model
        })

        for i in range(len(feature_names)):
            wd = wasserstein_distance(human_X_scaled[:, i], model_X_scaled[:, i])
            feature_results.append({
                "observed": wd,
                "dataset": cfg.dataset.name,
                "model": model,
                "feature": feature_names[i]
            })

    print("Calculate distance between model and model")
    model_model_results = []
    for i, model_a in enumerate(models):
        X_a = feature_scaler.transform(
            biber_features_dfs[biber_features_dfs["Model"] == model_a][feature_names].values
        )
        for model_b in models[i:]:
            X_b = feature_scaler.transform(
                biber_features_dfs[biber_features_dfs["Model"] == model_b][feature_names].values
            )
            mm_observed = compute_mmd2_biased_batched(X_a, X_b, sigma=sigma)[0].cpu().item()
            model_model_results.append({
                "observed": mm_observed,
                "dataset": cfg.dataset.name,
                "model_a": model_a,
                "model_b": model_b,
                "type": "Model-Model",
                "sigma": sigma.cpu().item()
            })

    model_model_df = pd.DataFrame(model_model_results)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(model_model_df.to_string())
        
    heatmap_df = model_model_df.pivot(index="model_a", columns="model_b", values="observed")
    heatmap_df = heatmap_df.combine_first(heatmap_df.T)

    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(heatmap_df.to_string())


    stability_results = pd.DataFrame(stability_results)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(stability_results.to_string(index=False))    

    mmd_results = pd.DataFrame(mmd_results)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(mmd_results.to_string(index=False))       

    mmd_ablation_spread = pd.DataFrame(mmd_ablation_spread)
    with pd.option_context("display.float_format", "{:0.8f}".format):   
        print(mmd_ablation_spread.to_string(index=False))

    feature_results = pd.DataFrame(feature_results)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(feature_results.to_string(index=False))

    variances = pd.DataFrame(variances)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(variances.to_string(index=False))


    print(f"Saved results to: {os.path.join(ARTEFACT_PATH, "model_model_mmd.csv")}")
    heatmap_df.to_csv(os.path.join(ARTEFACT_PATH, "model_model_mmd.csv"))

    print(f"Saved results to: {os.path.join(ARTEFACT_PATH, "stability_results.csv")}")
    stability_results.to_csv(os.path.join(ARTEFACT_PATH, "stability_results.csv"))
    
    print(f"Saved dataset results to: {os.path.join(ARTEFACT_PATH, "mmd_results.csv")}")
    mmd_results.to_csv(os.path.join(ARTEFACT_PATH, "mmd_results.csv"))
    
    print(f"Saved stability results to: {os.path.join(ARTEFACT_PATH, "feature_results.csv")}")
    feature_results.to_csv(os.path.join(ARTEFACT_PATH, "feature_results.csv"))

    print(f"Saved results to: {os.path.join(ARTEFACT_PATH, 'mmd_ablation_spread.csv')}")
    mmd_ablation_spread.to_csv(os.path.join(ARTEFACT_PATH, "mmd_ablation_spread.csv"))

    print(f"Saved results to {os.path.join(ARTEFACT_PATH, 'variances.csv')}")
    variances.to_csv(os.path.join(ARTEFACT_PATH, "variances.csv"))

    print("Get dimensional loadings")
    dimensional_loadings_df = pd.read_csv(os.path.join(ARTEFACT_PATH, "Dimensional_Loadings_1988.csv"), index_col=0)
    # Replace faulty Gemma_27B names 
    dimensional_loadings_df["group"] = dimensional_loadings_df["group"].str.replace(r"([A-z0-9_]*)_Zero_Shot_Gemma_27b", r"\1_Zero_Shot_Gemma_27B", regex=True)

    dimensonal_scaler = StandardScaler()
    human_dims_full = dimensional_loadings_df[dimensional_loadings_df["group"]=="Human_Full"].drop(["doc_ids", "group", "dataset"], axis=1).values
    dimensonal_scaler.fit(human_dims_full)
    human_dims_full_scaled = dimensonal_scaler.transform(human_dims_full)

    wasserstein_distances = {
        "dataset": [],
        "model": [],
        "method": [],

        "Distance_dim_1": [],
        "ci_lower_dim_1": [],
        "ci_upper_dim_1": [],
        "mean_dim_1": [],
        "std_dim_1": [],

        "Distance_dim_2": [],
        "ci_lower_dim_2": [],
        "ci_upper_dim_2": [],
        "mean_dim_2": [],
        "std_dim_2": [],

        "Distance_dim_3": [],
        "ci_lower_dim_3": [],
        "ci_upper_dim_3": [],
        "mean_dim_3": [],
        "std_dim_3": [],

        "Distance_dim_4": [],
        "ci_lower_dim_4": [],
        "ci_upper_dim_4": [],
        "mean_dim_4": [],
        "std_dim_4": [],

        "Distance_dim_5": [],
        "ci_lower_dim_5": [],
        "ci_upper_dim_5": [],
        "mean_dim_5": [],
        "std_dim_5": [],

        "Distance_dim_6": [],
        "ci_lower_dim_6": [],
        "ci_upper_dim_6": [],   
        "mean_dim_6": [],
        "std_dim_6": [],
    }

    print("Calculate human baseline")
    means, stds, cis_lower, cis_upper = human_human_wasserstein_baseline(human_dims_full_scaled, n=600, R=cfg.stat_eval.resamples, ci=cfg.stat_eval.CI, seed=cfg.run.seed)
    for d in range(human_dims_full_scaled.shape[1]):
            wasserstein_distances[f"Distance_dim_{d + 1}"].append(None)
            wasserstein_distances[f"ci_lower_dim_{d + 1}"].append(cis_lower[d])
            wasserstein_distances[f"ci_upper_dim_{d + 1}"].append(cis_upper[d])
            wasserstein_distances[f"mean_dim_{d + 1}"].append(means[d])
            wasserstein_distances[f"std_dim_{d + 1}"].append(stds[d])
            
    wasserstein_distances["dataset"].append(cfg.dataset.name)
    wasserstein_distances["model"].append("Human")
    wasserstein_distances["method"].append("Resample")
    
    # Z-Scale the 600 samples used in the study
    human_dims_scaled = dimensonal_scaler.transform(dimensional_loadings_df[dimensional_loadings_df["group"]=="Human"].drop(["doc_ids", "group", "dataset"], axis=1).values)
    df = dimensional_loadings_df[~dimensional_loadings_df["group"].isin(["Human", "Human_Few_Shot", "Human_Full"])]

    print("Calculating Wassersteind distance Human-Model")
    # For each model calculate for each dim the Wasserstein distance between human-model and the p-value of the observed Wasserstein distance
    for model in df["group"].unique():
        print(f"Calculating {model}")
        model_dist_z_scaled = dimensonal_scaler.transform(df[df["group"]==model].drop(["doc_ids", "group", "dataset"], axis=1).values)
        observed_wds, means, stds, lower_cis, upper_cis  = wasserstein_bootstrap(model_dist_z_scaled, human_dims_scaled)

        for d in range(model_dist_z_scaled.shape[1]):
            wasserstein_distances[f"Distance_dim_{d + 1}"].append(observed_wds[d])
            wasserstein_distances[f"ci_lower_dim_{d + 1}"].append(lower_cis[d])
            wasserstein_distances[f"ci_upper_dim_{d + 1}"].append(upper_cis[d])
            wasserstein_distances[f"mean_dim_{d + 1}"].append(means[d])
            wasserstein_distances[f"std_dim_{d + 1}"].append(stds[d])

        wasserstein_distances["dataset"].append(cfg.dataset.name)
        wasserstein_distances["model"].append(model)
        wasserstein_distances["method"].append("Bootstrap")
            
    wasserstein_distances_df = pd.DataFrame(wasserstein_distances)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(wasserstein_distances_df.to_string(index=False))

    print(f"Saved results to: {os.path.join(ARTEFACT_PATH, "wasserstein_distances.csv")}")
    wasserstein_distances_df.to_csv(os.path.join(ARTEFACT_PATH, "wasserstein_distances.csv"))
"""
    dataset_results = []
    results = []

    print("Load Biber dimensions")
    # Read in the biber dimensions of the dataset
    df_dimensions = pd.read_csv(os.path.join(ARTEFACT_PATH, "Dimensional_Loadings_1988.csv"), index_col=0)

    # Correct spelling misstake for one experiment
    df_dimensions["group"] = df_dimensions["group"].str.replace(r"([A-z0-9_]*)_Zero_Shot_Gemma_27b", r"\1_Zero_Shot_Gemma_27B", regex=True)
    human_full_X = df_dimensions[df_dimensions["group"] == "Human_Full"].drop(
        ["doc_ids", "group", "dataset"], axis=1
    ).values
    
    print("Get standard scaler on full human distribution")
    # Get the scalling for the full human distribution
    scaler = StandardScaler()
    scaler.fit(human_full_X)
    human_full_X_scaled = scaler.transform(human_full_X)
    print("Estiamte Kernel Bandwith on human distribution")
    # Estimate MMD^2 bandwith on full human distribution
    sigma = estimate_median_heuristic(
        scaler.transform(human_full_X_scaled)
    ).item()
    print(f"Kernel width of {sigma}")
    
    print(f"Get mean and std of MMD between samples from full human distribution of size {cfg.subsample.sample_size} for {cfg.stat_eval.resamples} resamples")
    stability_results = []
    for n in tqdm.tqdm(range(cfg.subsample.sample_size_lower, cfg.subsample.sample_size + 1, cfg.subsample.sample_size_step_size)):
        _, hh_mean, hh_std, ci_lower, ci_upper = human_human_mmd_baseline(human_full_X_scaled, sigma=sigma, n=n, R=cfg.stat_eval.resamples, ci=cfg.stat_eval.CI)
        stability_results.append({
            "n": n,
            "mean_score": hh_mean,
            "std_score": hh_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "dataset": cfg.dataset.name,
            "source": "Human_Full",
            "type": "Human-Human",
            "method": "Resample"
        })

   
    
    # 2) Prepare comparison dataframe (exclude Human_Full / Human_Few_Shot)
    df_comp = df_dimensions[~df_dimensions["group"].isin(["Human_Full", "Human_Few_Shot"])]
    dataset_results.append({
        "dataset": cfg.dataset.name,
        "hh_mean": hh_mean,
        "hh_std": hh_std,
        "sigma": sigma,
    })
    human_sample_ids = df_comp[df_comp["group"] == "Human"].sort_values(by=['doc_ids'])["doc_ids"]
    # Pre-compute scaled Human features once
    human_X = df_comp[df_comp["group"] == "Human"].sort_values(by=['doc_ids']).drop(
        ["doc_ids", "group", "dataset"], axis=1
    ).values
    human_X_scaled = scaler.transform(human_X)

    print("Loop over models")
    df_comp = df_comp[~df_comp["group"].isin(["Human"])] 
    # 3) Loop over models in model_list and compute MMDs
    for model in df_comp["group"].unique():
        try:
            print(f"Calculating ---- {model}")
            model_rows = df_comp[df_comp["group"] == model].sort_values(by=['doc_ids'])

            assert human_sample_ids.reset_index(drop=True).equals(model_rows.reset_index(drop=True)["doc_ids"])
            model_X = model_rows.drop(["doc_ids", "group", "dataset"], axis=1).values
            model_X_scaled = scaler.transform(model_X)
        
            observed_mmd, p_value = mmd_permutation_test(
                human_X_scaled,
                model_X_scaled,
                sigma=sigma, 
                R=cfg.stat_eval.resamples
            )
            scaled_effect = (observed_mmd - hh_mean) / hh_std

            # Get CI via bootstraping on model-human distribution
            ids = np.array(range(len(model_rows)))
            for n in tqdm.tqdm(range(cfg.subsample.sample_size_lower, ids.shape[0] + 1, cfg.subsample.sample_size_step_size)):
                scores = []
                for _ in range(cfg.subsample.sample_size_repetitions):
                    # Get the subsample set for the experiments
                    ids_bootstrap = np.random.choice(ids, size=n, replace=True)
                    scores.append(compute_mmd_unbiased_batched(human_X_scaled[ids_bootstrap], model_X_scaled[ids_bootstrap])[0].cpu())
                scores_np = np.array(scores)
                # Compute percentile CI
                ci = cfg.stat_eval.CI
                lower = np.percentile(scores_np, 0 + (100-ci)/2)
                upper = np.percentile(scores_np, 100 - (100-ci)/2)
                stability_results.append({
                "n": n,
                "mean_score": scores_np.mean(),
                "std_score": scores_np.std(),
                "ci_lower": lower,
                "ci_upper": upper,
                "dataset": cfg.dataset.name,
                "source": model,
                "type": "Human-Model",
                "method": "Bootstrap"
                })


            results.append({
                "model_group": model,
                "observed_mmd": observed_mmd,
                "mean_score": scores_np.mean(),
                "std_score": scores_np.std(),
                "ci_lower_scaled": (lower - hh_mean) / hh_std,
                "ci_upper_scaled": (upper - hh_mean) / hh_std,
                "p_value": p_value,
                "scaled_effect": scaled_effect,
                "dataset": cfg.dataset.name
            })
        except Exception as e:
            print(e)
            print(f"Skipping {model}")

    results_df = pd.DataFrame(results)

    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(results_df.to_string(index=False))    

    stability_df = pd.DataFrame(stability_results)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(stability_df.to_string(index=False))       

    dataset_df = pd.DataFrame(dataset_results)
    with pd.option_context("display.float_format", "{:0.8f}".format):
        print(dataset_df.to_string(index=False))

    print(f"Saved results to: {os.path.join(ARTEFACT_PATH, "Dimensional_Loadings_1988.csv")}")
    results_df.to_csv(os.path.join(ARTEFACT_PATH, "Result_Stats.csv"))
    
    print(f"Saved dataset results to: {os.path.join(ARTEFACT_PATH, "Dataset_Stats.csv")}")
    dataset_df.to_csv(os.path.join(ARTEFACT_PATH, "Dataset_Stats.csv"))
    
    print(f"Saved stability results to: {os.path.join(ARTEFACT_PATH, "Dataset_Stats.csv")}")
    stability_df.to_csv(os.path.join(ARTEFACT_PATH, "Stability_Results.csv"))

"""



@hydra.main(config_path=".././configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(f"Seed: {cfg.run.seed}")
    print(f"Datapath {cfg.datapath}")
    print(f"Dataset: {cfg.dataset.name}")
    get_statistical_analysis(cfg)

if __name__ == "__main__":
    main()


