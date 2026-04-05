# Model Search for Building Height Binary Classification 

This project implements an automated script for analyzing the best model for binary classification of building height. The script takes two configuration files as input: one for the data specifications and one with the models to be applied to the dataset. This script can handle one dataset at a time and multiple models simultaneously. 

The results are saved in timestamped directories *results/{DDMMYYYY_HHMMSS}/* containing:
- the performance evaluation .json file for each model with final model parameters selection and performance
- *accuracy_scores_comparison.csv*: stores all accuracy scores with the related standard deviations and confidence interval for each model 
- *result_bar_plot_{timestamp}.png*: visual bar plot representation with accuracy scores for each model
- *test_results_plots* directory, containing for each model a scatter plot comparing actual vs predicted building height classifications and a bar plot showing the breakdown of True Positives, True Negatives, False Positives, and False Negatives.
- models directory containing serialized ML models 
- copies of the configuration files used during execution (*data_config.json*, *models_config.json*)

> 📋 **Note**: The timestamped directories in results have been changed to more descriptive name

## Project Structure

```
height-classification/
├── configs/
│   ├── data_config.json        # Dataset generation configuration 
│   ├── models_config.json      # ML models and hyperparameters configuration
│   └── README.md               # Detailed configuration instructions
├── data/                       # Dataset files
├── logs/                       # Pipeline execution logs
├── notebooks/
│   └── data_preparation.ipynb  # Data exploration and preparation
├── results/                    # Experiment results (timestamped or descriptive directories)
│   ├── {experiment_name}/
│   │   ├── {model_name}_results.json
│   │   ├── key_metrics_scores_comparison.csv
│   │   ├── models/             # Serialized ML models
│   │   ├── test_results_plots/ # Actual vs predicted plots per model
│   │   ├── data_config.json    # Config copy used during execution
│   │   └── models_config.json  # Config copy used during execution
├── scripts/
│   ├── model_executor.py       # Core ML pipeline logic
│   ├── run_model_experiments.py# Experiment orchestration script
│   └── utils.py                # Utility functions
├── requirements.txt
└── run_experiments_for_height_classification.sh
```

### Prerequisites

- Python 3.8+
- Required packages:
    - install with: `pip install -r requirements.txt`

**Step 1: Configure the Pipeline**

Before running the pipeline, you need to set up two configuration files in the `configs/` directory:

- `data_config.json` - Defines the dataset to be used with the height cutoff threshold to build the target binary variable
- `models_config.json` - Specifies machine learning models and their hyperparameters

> 📋 **Note**: For detailed configuration instructions, see the README in the `configs/` directory.

**Step 2: Run the Pipeline**

Once your configuration files are ready, navigate to the project directory where the `run_experiments_for_height_classification.sh` script is located and execute the pipeline using the following command:

```bash
./run_experiments_for_height_classification.sh 
```