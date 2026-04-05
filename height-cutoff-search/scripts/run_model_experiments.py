"""Machine Learning Model Experiments Script.

This script executes ML experiments on multiple datasets using various
model configurations. It performs nested cross-validation, hyperparameter tuning,
and generates performance visualizations and summaries.

Usage:
    python run_model_experiments.py --model-config config/models.json --log-file experiments.log --results-directory results/ --data-directory data/
"""

from data_processing import DataProcessor
from model_executor import ModelExecutor
import os
import argparse
import json
from utils import (
    setup_logging,
    get_column_types,
    save_performance_key_indicators,
    display_mae_scores_heatmap,
    display_mae_scores_lineplot,
    print_mae_results,
)

import logging
import re
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Parse command line arguments for configuration and paths
    parser = argparse.ArgumentParser(
        description="Run height cutoff experiments with configuration file"
    )
    parser.add_argument(
        "--model-config",
        "-m",
        type=str,
        required=True,
        help="Path to model configuration file",
    )
    parser.add_argument("--log-file", type=str, required=True, help="Path to log file")
    parser.add_argument(
        "--results-directory",
        type=str,
        required=True,
        help="Path to results directory",
    )
    parser.add_argument("--data-directory", type=str, help="Path to data")
    args = parser.parse_args()

    # Configure logging system with file and console output
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING MODEL EXPERIMENTS ON DATASETS")
    logger.info("=" * 60)
    setup_logging(args.log_file)
    logger.info(f"Configuration file for model execution: {args.model_config}")

    # Setup directory paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_directory = args.results_directory
    models_directory = os.path.join(results_directory, "models")
    mae_scores_filename = "mae_scores_results.csv"

    # Determine the source data directory (provided in input or default results/data)
    if args.data_directory:
        data_directory = args.data_directory
    else:
        data_directory = os.path.join(results_directory, "data")

    # Initialize data processor for loading and splitting datasets
    berlin_file = os.path.join(
        project_root, "data", "processed", "dati_berlino_cleaned.csv"
    )
    data_processor = DataProcessor(berlin_file, data_directory)

    # Load and parse model configuration from .json file
    with open(args.model_config, "r") as file:
        model_config = json.load(file)
    logger.info("Model configuration file loaded")

    # Iterate through each model configuration, applying each model to
    # all datasets in the provided data directory
    for j, model in enumerate(model_config, 1):
        if not model.get("enable", True):
            logger.info(
                f"Skipping disabled model: "
                f"{model.get("experiment_name", f"Model: {j}")}"
            )
            continue

        for filename in os.listdir(data_directory):
            # Skip non-CSV files
            if not filename.endswith(".csv"):
                continue

            df_tmp = data_processor.load_data_from_csv(
                os.path.join(data_directory, filename)
            )

            # Extract dataset identifiers from filename using regex
            # Expected format: berlin_dataset_X_height_cutoff_Y_*.csv
            cutoff_value_match = re.search(r"cutoff_(\d+)", filename)
            cutoff_value = f"cutoff_{cutoff_value_match.group(1)}"
            dataset_name_match = re.search(r"dataset_(\d+)", filename)
            dataset_name = f"dataset_{dataset_name_match.group(1)}"

            model_experiment_name = model.get(
                "experiment_name", f"Model Experiment {j}"
            )
            logger.info("-" * 50)
            logger.info(
                f"MODEL EXPERIMENT {j}/{len(model_config)}: "
                f"{model_experiment_name} on dataset {dataset_name}"
            )
            logger.info("-" * 50)

            # Split dataset into training and testing datasets
            df_train, df_test = data_processor.test_train_data_split(df_tmp)

            # Cap values for the training dataset
            dual_evaluation_at_height_cutoff=model.get("dual_evaluation_at_height_cutoff", None)
            if dual_evaluation_at_height_cutoff:
                logger.info("capping height for the training dataset")
                df_train=data_processor.cap_values(df_train, dual_evaluation_at_height_cutoff-3,"height_of_fall_m")

            # Identify categorical and numerical columns for preprocessing pipeline step
            numerical_features, categorical_features = get_column_types(
                model["features"], df_tmp
            )

            # Initialize model executor and run complete ML experiment
            executor = ModelExecutor(model, results_directory, models_directory)
            results = executor.run_experiment(
                df_train,
                df_test,
                numerical_features,
                categorical_features,
                dataset_name,
            )

            # Extract and save performance estimation results from nested CV
            performance_estimation = results["performance_estimation"]
            mae_score_results = save_performance_key_indicators(
                results_directory,
                mae_scores_filename,
                model_experiment_name,
                f"{dataset_name}_{cutoff_value}",
                performance_estimation,
            )

            logger.info(
                f"Performance key indicators saved for "
                f"{mae_score_results["model_name"]} on {mae_score_results["dataset"]}, "
                f"MAE: {mae_score_results["mae"]}"
            )
            logger.info("-" * 50)
            logger.info(
                f"MODEL EXPERIMENT {j}/{len(model_config)} on dataset {dataset_name} "
                "ENDED SUCCESSFULLY"
            )
            logger.info("-" * 50)

    # Archive the configuration file in the experiment results directory
    config_file_path = os.path.join(results_directory, "model_config.json")
    with open(config_file_path, "w") as f:
        json.dump(model_config, f, indent=4)

    # Visualize and save plots with the entire experiment results
    mae_all_score_results = pd.read_csv(
        os.path.join(results_directory, mae_scores_filename)
    )
    print_mae_results(mae_all_score_results)

    file_lineplot_path = display_mae_scores_lineplot(
        results_directory, mae_all_score_results
    )
    logger.info(f"Lineplot saved at: {file_lineplot_path}")

    file_heatmap_path = display_mae_scores_heatmap(
        results_directory, mae_all_score_results
    )
    logger.info(f"Heatmap saved at: {file_heatmap_path}")

    plt.show()

    logger.info("=" * 60)
    logger.info("MODEL EXPERIMENTS ON DATASETS ENDED SUCCESSFULLY")
    logger.info("=" * 60)
