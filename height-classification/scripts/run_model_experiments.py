import os
import argparse
import json
import logging
from utils import (
    setup_logging,
    transform_column_to_binary_var,
    get_feature_lists,
    save_performance_key_indicators,
    print_accuracy_results,
    display_metric_scores_bar_plot,
)
import pandas as pd
from model_executor import ModelExecutor
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Parse arguments in input
    parser = argparse.ArgumentParser(
        description="Run height classification experiments with configuration file"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--log-file", type=str, required=True, help="Path to the log file"
    )
    parser.add_argument(
        "--results-directory",
        type=str,
        required=True,
        help="Path to the results for the experiment run",
    )
    parser.add_argument(
        "--script-directory",
        type=str,
        required=True,
        help="The directory from which the script is run",
    )

    args = parser.parse_args()
    script_directory = args.script_directory
    results_directory = args.results_directory
    data_config = args.data_config
    model_config = args.model_config

    key_metrics_all_results_filename = os.path.join(
        results_directory, "key_metrics_scores_comparison.csv"
    )

    # Logging setup
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING SETUP FOR EXPERIMENTS RUN")
    logger.info("=" * 60)

    # Load and Parse configuration files
    with open(data_config, "r") as f:
        data_config = json.load(f)

    with open(model_config, "r") as f:
        models = json.load(f)

    # Load dataset
    filename_data = data_config["filename_data"]
    data_path = os.path.join(script_directory, "data", filename_data)
    df = pd.read_csv(data_path)

    # Transform height into binary variable
    classification_height_threshold = data_config["height_classification_threshold"]
    df_with_binary_column, column_class_name = transform_column_to_binary_var(
        df, "height_of_fall_m", classification_height_threshold
    )

    # Get numerical and categorical feature lists for classification
    features_to_include = data_config["features"]
    numerical_features, categorical_features = get_feature_lists(
        features_to_include, df_with_binary_column.drop(columns=[column_class_name])
    )

    # Run model experiments
    for j, model in enumerate(models, 1):
        if not model.get("enable", True):
            logger.info(
                f"Skipping disabled model: "
                f"{model.get("experiment_name", f"Model: {j}")}"
            )
            continue

        model_experiment_name = model.get("experiment_name", f"Model Experiment {j}")
        logger.info("-" * 50)
        logger.info(f"MODEL EXPERIMENT {j}/{len(models)}: " f"{model_experiment_name}")
        logger.info("-" * 50)

        executor = ModelExecutor(model, results_directory)
        results = executor.run_experiment(
            df_with_binary_column,
            column_class_name,
            numerical_features,
            categorical_features,
        )

        # Extract and save performance estimation results from nested CV
        performance_estimation = results["performance_estimation"]
        key_metrics_results = save_performance_key_indicators(
            model_experiment_name, key_metrics_all_results_filename, performance_estimation
        )

        logger.info(
            f"Performance key indicators saved for "
            f"{key_metrics_results["model_name"]}, "
            f"Accuracy: {key_metrics_results["accuracy"]}"
        )

        logger.info("-" * 50)
        logger.info(f"MODEL EXPERIMENT {j}/{len(models)} " "ENDED SUCCESSFULLY")
        logger.info("-" * 50)

    # Save config files in the results directory
    model_config_file_path = os.path.join(results_directory, "models_config.json")
    with open(model_config_file_path, "w") as f:
        json.dump(models, f, indent=4)

    data_config_file_path = os.path.join(results_directory, "data_config.json")
    with open(data_config_file_path, "w") as f:
        json.dump(data_config, f, indent=4)

    # Visualize and save plots with the entire experiment results
    key_metrics_all_results = pd.read_csv(key_metrics_all_results_filename)
    print_accuracy_results(key_metrics_all_results)

    file_accuracy_barplot_path = display_metric_scores_bar_plot(
        results_directory, key_metrics_all_results, "accuracy"
    )
    logger.info(f"Bar Plot saved at: {file_accuracy_barplot_path}")

    file_f1_macro_barplot_path = display_metric_scores_bar_plot(
        results_directory, key_metrics_all_results, "f1_macro"
    )
    logger.info(f"Bar Plot saved at: {file_f1_macro_barplot_path}")

    plt.show()

    logger.info("=" * 60)
    logger.info("MODEL EXPERIMENTS ENDED SUCCESSFULLY")
    logger.info("=" * 60)
