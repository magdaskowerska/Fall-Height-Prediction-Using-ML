"""Dataset Generation Script for Machine Learning Experiments.

This script processes raw data through various transformation steps to generate
multiple datasets for ML experiments. It supports operations like outlier removal,
value capping, and height normalization based on JSON configuration files.

Usage:
    python run_datasets_generation.py --data-config config/data_config.json --results-directory results/
"""

from data_processing import DataProcessor
import os
import argparse
import json
from utils import setup_logging
import logging

if __name__ == "__main__":
    # Parse command line arguments for configuration and output paths
    parser = argparse.ArgumentParser(
        description="Run height cutoff experiments with configuration file"
    )
    parser.add_argument(
        "--data-config",
        "-d",
        type=str,
        required=True,
        help="Path to data processing configuration file",
    )
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument(
        "--results-directory", type=str, help="Path to results directory"
    )
    args = parser.parse_args()

    # Configure logging system with file and console output
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING DATASETS GENERATION")
    logger.info("=" * 60)
    logger.info(f"Configuration file for data preparation: {args.data_config}")

    # Setup directory paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    berlin_file = os.path.join(
        project_root, "data", "processed", "dati_berlino_cleaned.csv"
    )

    results_directory = args.results_directory
    processed_data_directory = os.path.join(results_directory, "data")

    # Initialize data processor and load base dataset ready for transformations
    data_processor = DataProcessor(berlin_file, processed_data_directory)
    df = data_processor.load_data_from_csv()

    # Load and parse data processing configuration .json file
    with open(args.data_config, "r") as file:
        data_config = json.load(file)
    logger.info("Configuration file for data processing loaded")

    # Process each dataset configuration sequentially
    for i, item in enumerate(data_config, 1):

        data_experiment_name = item.get(
            "experiment_name", f"Data Processing Experiment {i}"
        )
        logger.info("-" * 50)
        logger.info(
            f"DATA PROCESSING EXPERIMENT {i}/{len(data_config)}: {data_experiment_name}"
        )
        logger.info("-" * 50)

        df_tmp = df.copy()
        # Track applied processing steps for filename
        processing_steps = [f"dataset_{i}"]

        # Remove outliers if configured
        if remove_outliers := item.get("remove_outliers"):
            column_name = remove_outliers["column_name"]
            threshold = remove_outliers.get("threshold", 1.5)
            df_tmp = data_processor.remove_outliers(df, column_name, threshold)
            processing_steps.append("outliers_removed")

        # Cap or remove extreme values if configured
        if cap_column_value := item.get("cap_column_values"):
            column_name = cap_column_value["column_name"]
            cutoff_value = cap_column_value["cutoff_value"]
            method = cap_column_value.get("method", "cap")
            df_tmp = data_processor.cap_values(
                df_tmp, cutoff_value, column_name, method
            )
            processing_steps.append(f"height_cutoff_{cutoff_value}")

        # Normalize height values to floor levels if configured
        if item.get("normalization") == "True":
            df_tmp = data_processor.normalize_height_values(df_tmp)
            processing_steps.append("normalized_height")

        # Save processed dataset
        data_processor.save_data_to_csv(df_tmp, processing_steps)

    # Archive the configuration file in the experiment results directory
    config_file_path = os.path.join(results_directory, "height_cutoff_config.json")
    with open(config_file_path, "w") as f:
        json.dump(data_config, f, indent=4)

    logger.info("=" * 60)
    logger.info("DATASETS GENERATION ENDED SUCCESSFULLY")
    logger.info("=" * 60)
