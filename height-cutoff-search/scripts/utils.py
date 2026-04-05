import logging
import os
from datetime import datetime
from typing import Dict, Type, Tuple, List, Any
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import re
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

# Component mappings for dynamic class loading
MODEL_MAPPING: Dict[str, Type] = {
    "LinearRegression": LinearRegression,
    "BayesianRidge": BayesianRidge,
    "SVR": SVR,
    "KernelRidge": KernelRidge,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}

PREPROCESSOR_MAPPING: Dict[str, Type] = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "OrdinalEncoder": OrdinalEncoder,
}

DIMENSIONALITY_REDUCTION: Dict[str, Type] = {
    "PCA": PCA,
    "SVD": TruncatedSVD,
    "ICA": FastICA,
}

CV_MAPPING: Dict[str, Type] = {
    "KFold": KFold,
    "StratifiedKFold": StratifiedKFold,
    "TimeSeriesSplit": TimeSeriesSplit,
}


def setup_logging(
    log_file_path: str = None,
    log_level: int = logging.INFO,
    log_file_flag: bool = True,
    console_output: bool = True,
):
    """Configure logging system with file and console output.

    Args:
        log_file_path (Optional[str]): path to log file.
            If None, generates timestamped filename
        log_level (int): logging level (default: logging.INFO)
        log_file_flag (bool): enable file logging (default: True)
        console_output (bool): enable console logging (default: True)
    """
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"logs/{timestamp}.log"

    if log_file_flag:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s - %(name)-15s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = []

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)

    if log_file_flag:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )


def get_column_types(
    features_to_include: str,
    df: DataFrame,
    y_column: str = "height_of_fall_m",
) -> Tuple[List[str], List[str]]:
    """Classify DataFrame columns into numerical and categorical categories.

    Args:
        features_to_include (str): feature selection strategy:
            - 'no_bin': features without binary variables
            - 'no_bin_no_agg': only demographic features
            - 'all': all features in the dataset
            - 'bin': all numerical except anatomical region features
        df (DataFrame): input DataFrame representing the entire dataset
            for the ML experiment
        y_column (str): target column name to exclude (default: 'height_of_fall_m')

    Returns:
        numerical_features, categorical_features (Tuple[List[str], List[str]]):
            lists of numerical and categorical column names
    """

    df_copy = df.drop(columns=[y_column])
    categorical_features = df_copy.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if features_to_include == "no_bin":
        numerical_features = [
            "age",
            "subject_height",
            "weight",
            "bmi",
            "head",
            "thorax",
            "skeleton",
            "abdomen",
        ]

    elif features_to_include == "no_bin_no_agg":
        numerical_features = ["age", "subject_height", "weight", "bmi"]

    elif features_to_include == "all":
        numerical_features = df_copy.select_dtypes(include=["number"]).columns.tolist()

    elif features_to_include == "bin":
        all_numerical_features = df_copy.select_dtypes(
            include=["number"]
        ).columns.tolist()

        excluded_anatomical_features = {
            "head",
            "thorax",
            "skeleton",
            "abdomen",
        }

        numerical_features = [
            feature
            for feature in all_numerical_features
            if feature not in excluded_anatomical_features
        ]

    return numerical_features, categorical_features


def save_performance_key_indicators(
    directory: str,
    filename: str,
    model_name: str,
    dataset_name: str,
    performance_estimation: Dict[str, Any],
) -> Dict[str, Any]:
    """Save and update model performance key indicators in a CSV summary file.

    Args:
        directory (str): directory path for saving the performance file
        filename (str): name of the CSV file to save/update
        model_name (str): name identifier of the ML model
        dataset_name (str): name identifier of the dataset
        performance_estimation (Dict[str, Any]):
            nested CV performance results containing:
            - nested_cv.neg_mean_absolute_error.scores_mean
            - nested_cv.neg_mean_absolute_error.scores_std
            - nested_cv.neg_mean_absolute_error.confidence_interval.margin_of_error

    Returns:
        Dict[str, Any]: dictionary of the saved entry with keys:
            - model_name, dataset, mae, std, confidence_interval_mae
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    result_path = os.path.join(directory, filename)

    if os.path.exists(result_path):
        df_performance_key_indicators = pd.read_csv(result_path)
    else:
        df_performance_key_indicators = DataFrame(
            columns=[
                "model_name",
                "dataset",
                "mae",
                "std",
                "confidence_interval_mae",
            ]
        )

    mae = abs(
        performance_estimation["nested_cv"]["neg_mean_absolute_error"]["scores_mean"]
    )
    std = performance_estimation["nested_cv"]["neg_mean_absolute_error"]["scores_std"]
    margin_of_error = performance_estimation["nested_cv"]["neg_mean_absolute_error"][
        "confidence_interval"
    ]["margin_of_error"]

    new_entry = DataFrame(
        {
            "model_name": [model_name],
            "dataset": [dataset_name],
            "mae": [mae],
            "std": [std],
            "confidence_interval_mae": [f"{mae:.4f} ±{margin_of_error:.4f}"],
        }
    )

    df_performance_key_indicators = df_performance_key_indicators[
        ~(
            (df_performance_key_indicators["model_name"] == model_name)
            & (df_performance_key_indicators["dataset"] == dataset_name)
        )
    ]
    df_performance_key_indicators = pd.concat(
        [df_performance_key_indicators, new_entry], ignore_index=True
    )

    try:
        df_performance_key_indicators.to_csv(result_path, index=False)
    except Exception as e:
        raise ValueError(f"The file cannot be saved at path {result_path}: {e}")

    return new_entry.iloc[0].to_dict()


def display_mae_scores_heatmap(result_directory: str, all_mae_scores: DataFrame) -> str:
    """Generate and save a heatmap visualization of MAE scores ± standard
    deviation across models and datasets.

    Args:
        result_directory (str): directory path for saving the plot
        all_mae_scores (DataFrame): performance data with columns:
            - model_name: ML model identifier
            - dataset: dataset identifier (must contain 'cutoff_X' pattern)
            - mae: mean absolute error values
            - std: standard deviation values

    Returns:
        plot_path (str): path to the saved heatmap image file
    """
    custom_colors = ["#147514", "#25DA25", "#9BEE9B"]
    n_bins = 100
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        "custom", custom_colors, N=n_bins
    )

    df = all_mae_scores.copy()
    df["cutoff_value"] = df["dataset"].apply(
        lambda x: int(re.search(r"cutoff_(\d+)", x).group(1))
    )
    df = df.sort_values("cutoff_value")

    mae_all_scores_table = df.pivot_table(
        values="mae", index="model_name", columns="dataset", aggfunc="first"
    )

    mae_all_std_table = df.pivot_table(
        values="std", index="model_name", columns="dataset", aggfunc="first"
    )

    dataset_cutoff_mapping = df.set_index("dataset")["cutoff_value"].to_dict()

    sorted_columns = sorted(
        mae_all_scores_table.columns, key=lambda x: dataset_cutoff_mapping[x]
    )
    mae_all_scores_table = mae_all_scores_table[sorted_columns]
    mae_all_std_table = mae_all_std_table[sorted_columns]

    combined_annotations = pd.DataFrame(
        index=mae_all_scores_table.index,
        columns=mae_all_scores_table.columns,
        dtype=object
    )
    
    for i in range(len(mae_all_scores_table.index)):
        for j in range(len(mae_all_scores_table.columns)):
            mae_val = mae_all_scores_table.iloc[i, j]
            std_val = mae_all_std_table.iloc[i, j]
            combined_annotations.iloc[i, j] = f"{mae_val:.3f}\n±{std_val:.3f}"

    n_models = len(mae_all_scores_table.index)
    n_datasets = len(mae_all_scores_table.columns)
    fig_width = max(10, min(16, n_datasets * 3))
    fig_height = max(6, min(12, n_models * 1.5)) 

    plt.figure(figsize=(fig_width, fig_height))

    sns.heatmap(
        mae_all_scores_table,
        annot=combined_annotations,
        fmt="",
        cmap=custom_cmap,
        cbar_kws={
            "label": "Mean Absolute Error (MAE)",
            "pad": 0.1,
            "aspect": 15,
            "shrink": 0.8,
            "fraction": 0.04,
        },
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10 if n_models <= 3 else 8},
    )

    plt.title("Model Performance Comparison Across Datasets", fontsize=14, pad=50)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=3.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"results_heatmap_{timestamp}.png"
    plot_path = os.path.join(result_directory, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    return plot_path


def display_mae_scores_lineplot(
    result_directory: str, all_mae_scores: DataFrame, std_flag: bool=True
) -> str:
    """Generate and save a line plot showing MAE performance trends across
    height cutoff values, with error bars representing standard deviations.

    Args:
        result_directory (str): directory path for saving the plot
        all_mae_scores (DataFrame): performance data with columns:
            - model_name: ML model identifier
            - dataset: Dataset identifier (must contain 'cutoff_X' pattern)
            - mae: Mean absolute error values
            - std: Standard deviation values

    Returns:
        plot_path (str): Full path to the saved line plot image file
    """
    df = all_mae_scores.copy()

    df["cutoff_value"] = df["dataset"].apply(
        lambda x: int(re.search(r"cutoff_(\d+)", x).group(1))
    )

    plt.figure(figsize=(9, 6))
    unique_models = df["model_name"].unique()
    colors = plt.cm.Set1(range(len(unique_models)))

    for i, model in enumerate(unique_models):
        model_data = df[df["model_name"] == model].copy()
        model_data = model_data.sort_values("cutoff_value")

        if std_flag:
            plt.errorbar(
                model_data["cutoff_value"],
                model_data["mae"],
                yerr=model_data["std"],
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                label=model,
                color=colors[i],
                alpha=0.8,
            )
        else:
            plt.plot(model_data['cutoff_value'], model_data['mae'], 
                marker='o', linewidth=2, markersize=6,
                label=model, color=colors[i])

        for _, row in model_data.iterrows():
            plt.annotate(
                f'{row["mae"]:.3f}',
                (row["cutoff_value"], row["mae"]),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    plt.xlabel("Height Cutoff Value (m)", fontsize=12)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
    plt.title("Model Performance vs Height Cutoff Values", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    cutoff_values = sorted(df["cutoff_value"].unique())
    plt.xticks(cutoff_values)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"mae_vs_cutoff_lineplot_{timestamp}.png"
    plot_path = os.path.join(result_directory, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    return plot_path


def save_test_final_model_plot(
    y_test: np.ndarray, y_pred: np.ndarray, plot_test_final_model_path: str
) -> str:
    """Generate and save a scatter plot comparing actual vs predicted values
    for model evaluation.

    Args:
        y_test (np.ndarray): actual target values from test set
        y_pred (np.ndarray): predicted values from the model
        plot_test_final_model_path (str): full path where plot should be saved

    Returns:
        plot_test_final_model_path (str): path to the saved plot file
    """
    if not os.path.exists(os.path.dirname(plot_test_final_model_path)):
        os.makedirs(os.path.dirname(plot_test_final_model_path), exist_ok=True)

    fig = plt.figure(figsize=(14, 8))

    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)

    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot(y_test, p(y_test), "r--", label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    line_x = np.linspace(min_val, max_val, 100)
    line_y = line_x
    plt.plot(
        line_x,
        line_y,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="Perfect Prediction",
    )

    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values (y_test)")
    plt.ylabel("Predicted Values (y_pred)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        plot_test_final_model_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return plot_test_final_model_path


def print_mae_results(mae_all_score_results: DataFrame):
    """Display model performance results as a table on the console.

    Args:
        mae_all_score_results (DataFrame): performance results with columns:
            - model_name: ML model identifier
            - dataset: Dataset identifier
            - mae: mean absolute error
            - std: standard deviation
            - confidence_interval_mae: confidence interval string
    """
    console = Console()

    table = Table(title="Model Performance Results", box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dataset", style="magenta")
    table.add_column("MAE", style="green", justify="right")
    table.add_column("Std", style="yellow", justify="right")
    table.add_column("Confidence Interval", style="blue")

    for _, row in mae_all_score_results.sort_values("mae").iterrows():
        table.add_row(
            row["model_name"],
            row["dataset"],
            f"{row['mae']:.4f}",
            f"{row['std']:.4f}",
            row["confidence_interval_mae"],
        )

    console.print(table)

    return


def get_component_class(mapping: Dict[str, Type], name: str) -> Type:
    """Generic function to retrieve a class from a mapping dictionary.

     Args:
        mapping (Dict[str, Type]): dictionary mapping names to class types
        name (str): name of the component to retrieve

    Returns:
        mapping[name] (Type): the requested class type
    """
    if name not in mapping:
        raise ValueError(f"Model {name} not available in the mapping: {mapping}")
    return mapping[name]


def get_model_class(name: str) -> Type:
    """Get model class by name."""
    return get_component_class(MODEL_MAPPING, name)


def get_preprocessor_class(name: str) -> Type:
    """Get preprocessor class by name."""
    return get_component_class(PREPROCESSOR_MAPPING, name)


def get_dim_reducer_class(name: str) -> Type:
    """Get dimensionality reducer class by name."""
    return get_component_class(DIMENSIONALITY_REDUCTION, name)


def get_cv_strategy_class(name: str) -> Type:
    """Get CV strategy class by name."""
    return get_component_class(CV_MAPPING, name)


def save_test_performance_binary_var_plot(
    tp: int, tn: int, fp: int, fn: int, plot_test_final_model_path: str
) -> str:
    """Generate and save a bar plot showing the number of True Positives,
    True Negatives, False Positives, and False Negatives.

    Args:
        tp (int): true positives
        tn (int): true negatives
        fp (int): false positives
        fn (int): false negatives
        plot_test_final_model_path (str): full path where plot should be saved

    Returns:
        plot_test_final_model_path (str): path to the saved plot file
    """
    if not os.path.exists(os.path.dirname(plot_test_final_model_path)):
        os.makedirs(os.path.dirname(plot_test_final_model_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = [
        "True\nPositives",
        "True\nNegatives",
        "False\nPositives",
        "False\nNegatives",
    ]
    values = [tp, tn, fp, fn]
    colors_bar = ["lightgreen", "lightblue", "lightcoral", "lightyellow"]

    bars = ax.bar(categories, values, color=colors_bar, edgecolor="black")
    ax.set_title("Height Classification")
    ax.set_ylabel("Count")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    plt.savefig(
        plot_test_final_model_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return plot_test_final_model_path