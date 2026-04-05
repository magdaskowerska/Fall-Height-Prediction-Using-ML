# Height Cutoff Analysis for Height of Fall Prediction

This project implements an automated pipeline for analyzing the impact of height cutoffs on building height prediction accuracy. The pipeline is divided into two main steps:
1. Data Generation: generates the datasets according to the configuration file *height_cutoff_config.json*
2. Model Experiment: applies the machine learning models defined in the configuration file *models_config.json* to the generated datasets

The results are saved in timestamped directories *results/{DDMMYYYY_HHMMSS}/* containing:
- the performance evaluation .json file for each model and dataset with final model parameters selection and performance
- *mae_scores_results.csv*: stores all MAE scores with the related standard deviations and confidence interval for each model and dataset combinations
- *result_heatmap_{timestamp}.png*: visual heatmap representation of MAE scores across all experiments 
- *mae_vs_cutoff_lineplot_{timestamp}.png*: line plot showing MAE vs height cutoff values trend
- copies of the configuration files used during execution (*height_cutoff_config.json*, *models_config.json*)
- *outliers_predicted_y* directory, containing .csv files with records where the prediction residuals (difference between y predicted and its actual value) exceed a threshold
- *plots_pred_vs_real_y* directory, containing line plots comparing actual vs predicted height values
- models directory containing serialized ML models for each experiment
- data directory containing processed datasets

> 📋 **Note**: The timestamped directories in results have been changed to more descriptive name

## Project Structure

```
height-cutoff-search/
├── configs/
│   ├── height_cutoff_config.json   # Dataset generation configuration (height cutoffs, outliers)
│   ├── models_config.json          # ML models and hyperparameters configuration
│   └── README.md                   # Detailed configuration instructions
├── data/
│   ├── raw/                        # Raw input dataset files
│   └── processed/                  # Generated datasets per height cutoff
├── logs/                           # Pipeline execution logs
├── notebooks/
│   ├── exploratory_analysis.ipynb          # Data exploration and analysis
│   └── final_results_visualization.ipynb   # Results visualization
├── results/                        # Experiment results (timestamped or descriptive directories)
│   ├── {experiment_name}/
│   │   ├── {model_name}_{dataset_name}_results.json
│   │   ├── mae_scores_results.csv
│   │   ├── result_heatmap_{timestamp}.png
│   │   ├── mae_vs_cutoff_lineplot_{timestamp}.png
│   │   ├── models/                 # Serialized ML models
│   │   ├── data/                   # Processed datasets
│   │   ├── outliers_predicted_y/   # Prediction residual outlier files
│   │   ├── plots_pred_vs_real_y/   # Actual vs predicted plots
│   │   ├── height_cutoff_config.json   # Config copy used during execution
│   │   └── models_config.json          # Config copy used during execution
├── scripts/
│   ├── data_processing.py          # Dataset generation and preprocessing
│   ├── model_executor.py           # Core ML pipeline logic
│   ├── run_model_experiments.py    # Experiment orchestration script
│   └── utils.py                    # Utility functions
├── requirements.txt
└── run_experiments_for_height_cutoff.sh
```

### Prerequisites

- Python 3.8+
- Required packages:
    - install with: `pip install -r requirements.txt`

### Usage

**Step 1: Configure the Pipeline**

Before running the pipeline, you need to set up two configuration files in the `configs/` directory:

- `height_cutoff_config.json` - Defines dataset generation parameters (height cutoffs, remove outlier, etc.)
- `models_config.json` - Specifies machine learning models and their hyperparameters

> 📋 **Note**: For detailed configuration instructions, see the README in the `configs/` directory.

**Step 2: Run the Pipeline**

Once your configuration files are ready, navigate to the project directory where the `run_experiments_for_height_cutoff.sh` script is located and execute the pipeline using one of the following commands:

```bash
# Run complete pipeline
./run_experiments_for_height_cutoff.sh --both

# Run only data generation
./run_experiments_for_height_cutoff.sh --data-generation

# Run only model experiments (requires existing processed datasets)
# DATA_DIRECTORY: path to the directory containing processed data in the results directory, ex: '18012026_102911/data'
./run_experiments_for_height_cutoff.sh --models-experiment [DATA_DIRECTORY]
```