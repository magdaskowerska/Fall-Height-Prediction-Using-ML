import logging
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Type, Any
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OrdinalEncoder,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime


# Component mappings for dynamic class loading
MODEL_MAPPING: Dict[str, Type] = {
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "GaussianNB": GaussianNB,
    "KNeighborsClassifier": KNeighborsClassifier,
    "XGBClassifier": XGBClassifier,
    "RandomForestClassifier": RandomForestClassifier,
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


def transform_column_to_binary_var(
    df: pd.DataFrame, column_name: str, classification_threshold: int
) -> Tuple[pd.DataFrame, str]:
    """
    Transform column to a binary variable taking values 1 when higher than the specified
    threshold otherwise the variable is set to 0.

    Args:
        - df (DataFrame): pandas DataFrame containing the column to transform
        - column_name (str): column name to be transformed
        - classification_threshold (int): the threshold used for the transformation

    Returns:
        - df, column_class_name (Tuple[pd.DataFrame, str]):
            the pandas dataframe with the specified column
            transformed to binary variable and the new column name

    """

    column_class_name = "height_class"
    df[column_class_name] = np.where(df[column_name] > classification_threshold, 1, 0)

    df = df.drop(columns=[column_name])

    return df, column_class_name


def setup_logging(
    log_file_path: str = None,
    log_level: int = logging.INFO,
    log_file_flag: bool = True,
    console_output: bool = True,
):
    """Configure logging system with file and console output.

    Args:
        log_file_path (Optional[str]): path to log file.
            Mandatory with log_file_flag is set to True.
        log_level (int): logging level (default: logging.INFO)
        log_file_flag (bool): enable file logging (default: True)
        console_output (bool): enable console logging (default: True)
    """

    if log_file_flag and (log_file_path is None):
        logging.error("Path to log file empty, please provide a path for saving logs.")
        raise ValueError(
            "Path to log file empty, please provide a path for saving logs."
        )

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


def get_feature_lists(
    features_to_include: str,
    df: pd.DataFrame,
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

    Returns:
        numerical_features, categorical_features (Tuple[List[str], List[str]]):
            lists of numerical and categorical column names
    """

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    anatomical_grouped_features = ["head", "thorax", "skeleton", "abdomen"]

    demographic_features = ["age", "subject_height", "weight", "bmi"]

    if features_to_include == "no_bin":
        numerical_features = anatomical_grouped_features + demographic_features
    elif features_to_include == "no_bin_no_agg":
        numerical_features = demographic_features
    elif features_to_include == "all":
        numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
    elif features_to_include == "bin":
        numerical_features_all = df.select_dtypes(include=["number"]).columns.tolist()

        numerical_features = [
            feature
            for feature in numerical_features_all
            if feature not in anatomical_grouped_features
        ]

    return numerical_features, categorical_features


def save_performance_key_indicators(
    model_name: str,
    path_to_file: str,
    performance_estimation: Dict[str, Any],
) -> Dict[str, Any]:
    """Save and update model performance key indicators in a CSV summary file.

    Args:
        path_to_file (str): path to the CSV file to save
        model_name (str): name identifier of the ML model
        performance_estimation (Dict[str, Any]):
            nested CV performance results containing:
            - nested_cv.accuracy.scores_mean
            - nested_cv.accuracy.scores_std
            - nested_cv.accuracy.confidence_interval.margin_of_error

    Returns:
        Dict[str, Any]: dictionary of the saved entry with keys:
            - model_name, accuracy, std, confidence_interval_accuracy
    """

    if os.path.exists(path_to_file):
        df_performance_key_indicators = pd.read_csv(path_to_file)
    else:
        df_performance_key_indicators = pd.DataFrame(
            columns=[
                "model_name",
                "accuracy",
                "accuracy_std",
                "f1_macro",
                "f1_macro_std"
            ]
        )

    accuracy = abs(performance_estimation["nested_cv"]["accuracy"]["scores_mean"])
    accuracy_std = performance_estimation["nested_cv"]["accuracy"]["scores_std"]
    f1_macro = abs(performance_estimation["nested_cv"]["f1_macro"]["scores_mean"])
    f1_macro_std = performance_estimation["nested_cv"]["f1_macro"]["scores_std"]
    

    new_entry = pd.DataFrame(
        {
            "model_name": [model_name],
            "accuracy": [accuracy],
            "accuracy_std": [accuracy_std],
            "f1_macro": [f1_macro],
            "f1_macro_std": [f1_macro_std]
        }
    )

    if df_performance_key_indicators.empty:
        df_performance_key_indicators = new_entry
    else:
        df_performance_key_indicators = pd.concat(
            [df_performance_key_indicators, new_entry], ignore_index=True
        )

    try:
        df_performance_key_indicators.to_csv(path_to_file, index=False)
    except Exception as e:
        raise ValueError(f"The file cannot be saved at path {path_to_file}: {e}")

    return new_entry.iloc[0].to_dict()


def save_test_final_model_plot(
    y_true: np.ndarray, y_pred: np.ndarray, plot_test_final_model_path: str
) -> str:
    """Generate and save a scatter plot comparing actual vs predicted values
    for model evaluation and bar plot showing the number of True Negatives,
    True Positives, False Negatives, False Positives.

    Args:
        y_true (np.ndarray): actual target values from test set
        y_pred (np.ndarray): predicted values from the model
        plot_test_final_model_path (str): full path where plot should be saved

    Returns:
        plot_test_final_model_path (str): path to the saved plot file
    """
    if not os.path.exists(os.path.dirname(plot_test_final_model_path)):
        os.makedirs(os.path.dirname(plot_test_final_model_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Scatter plot with jitter
    jitter_strength = 0.1
    x_jittered = y_true + np.random.normal(0, jitter_strength, len(y_true))
    y_jittered = y_pred + np.random.normal(0, jitter_strength, len(y_pred))

    axes[0].scatter(x_jittered, y_jittered, alpha=0.6, s=50)
    axes[0].plot([0, 1], [0, 1], "r--", lw=2, label="Perfect Prediction")
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].set_ylim(-0.3, 1.3)
    axes[0].set_xlabel("True Height")
    axes[0].set_ylabel("Predicted Height")
    axes[0].set_title("Predicted vs True (with jitter)")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Low", "High"])
    axes[0].set_yticklabels(["Low", "High"])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right plot: Classification breakdown bar chart
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    categories = [
        "True\nNegatives",
        "False\nPositives",
        "False\nNegatives",
        "True\nPositives",
    ]
    values = [tn, fp, fn, tp]
    colors_bar = ["lightblue", "lightcoral", "lightyellow", "lightgreen"]

    bars = axes[1].bar(categories, values, color=colors_bar, edgecolor="black")
    axes[1].set_title("Height Classification")
    axes[1].set_ylabel("Count")

    for bar, value in zip(bars, values):
        axes[1].text(
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


def display_metric_scores_bar_plot(
    result_directory: str, all_metric_scores: pd.DataFrame, metric: str
) -> str:
    """Generate and save a bar plot visualization of Accuracy scores ± standard
    deviation across models.

    Args:
        result_directory (str): directory path for saving the plot
        all_metric_scores (DataFrame): performance data with columns:
            - model_name: ML model identifier
            - accuracy: accuracy values
            - accuracy_std: accuracy standard deviation values 
            - f1_macro: f1_macro values
            - f1_macro_std: f1 macro standard deviation values

    Returns:
        plot_path (str): path to the saved bar plot image file
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(all_metric_scores))
    n_models = len(all_metric_scores)
    cmap = plt.cm.Set3
    colors = [cmap(i / n_models) for i in range(n_models)]

    bars = ax.barh(
        y_pos,
        all_metric_scores[metric],
        xerr=all_metric_scores[f"{metric}_std"],
        color=colors,
        alpha=0.7,
        capsize=5,
        edgecolor="black",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_metric_scores["model_name"])
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(
        "Building Height Classification - Model Performance",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, 1)

    for i, (bar, acc) in enumerate(zip(bars, all_metric_scores[metric])):
        ax.text(
            bar.get_width() + all_metric_scores[f"{metric}_std"].iloc[i] + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}±{all_metric_scores[f"{metric}_std"].iloc[i]:.3f}",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{metric}_results_bar_plot_{timestamp}.png"
    plot_path = os.path.join(result_directory, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")

    return plot_path


def print_accuracy_results(accuracy_all_score_results: pd.DataFrame):
    """Display model performance results as a table on the console.

    Args:
        accuracy_all_score_results (DataFrame): performance results with columns:
            - model_name: ML model identifier
            - accuracy: accuracy values
            - accuracy_std: accuracy standard deviation
            - f1_macro: f1_macro values
            - f1_macro_std: f1_macro standard deviation
            - confidence_interval_accuracy: confidence interval string
    """
    console = Console()

    table = Table(title="Model Performance Results", box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Accuracy Std", style="yellow", justify="right")
    table.add_column("F1 Macro", style="green", justify="right")
    table.add_column("F1 Macro Std", style="yellow", justify="right")

    for _, row in accuracy_all_score_results.sort_values("accuracy").iterrows():
        table.add_row(
            row["model_name"],
            f"{row['accuracy']:.4f}",
            f"{row['accuracy_std']:.4f}",
            f"{row['f1_macro']:.4f}",
            f"{row['f1_macro_std']:.4f}",
        )

    console.print(table)

    return

def apply_dataset_undersampling(
    df: pd.DataFrame, 
    target_column: str, 
    sampling_strategy: str = 'auto',
    random_state: int = 42,
    preserve_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply undersampling to balance dataset and return discarded samples for blind testing.
    
    Args:
        df (DataFrame): input dataset
        target_column (str): name of the target column
        random_state (int): random seed for reproducibility
        sampling_strategy (str): sampling strategy
        preserve_ratio (float): fraction of minority class to 
            remove from training and add to discarded dataset
        
    Returns:
        Tuple[DataFrame, DataFrame]: 
            - df_balanced: balanced dataset for training/testing
            - df_discarded: discarded majority class samples for blind testing
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    class_counts = y.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    logging.info(f"Original class distribution:")
    logging.info(f"  Class {minority_class}: {class_counts[minority_class]} samples")
    logging.info(f"  Class {majority_class}: {class_counts[majority_class]} samples")

    if sampling_strategy=='custom':
        minority_count = class_counts[minority_class]
        minority_preserve_count = int(minority_count * preserve_ratio)
        minority_for_training = minority_count - minority_preserve_count
        
        custom_sampling_strategy = {
            majority_class: minority_for_training, 
            minority_class: minority_for_training 
        }
        sampling_strategy=custom_sampling_strategy
    
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    
    kept_indices = set(undersampler.sample_indices_)
    all_indices = set(df.index)
    discarded_indices = all_indices - kept_indices
    
    df_balanced = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name=target_column)
    ], axis=1).reset_index(drop=True)
    
    df_discarded = df.loc[list(discarded_indices)].reset_index(drop=True)
    
    balanced_class_counts = pd.Series(y_resampled).value_counts()
    discarded_class_counts = df_discarded[target_column].value_counts()
    
    logging.info(f"After undersampling:")
    logging.info(f"Balanced dataset: {len(df_balanced)} samples")
    for class_val, count in balanced_class_counts.items():
        logging.info(f"  Class {class_val}: {count} samples")
    
    logging.info(f"Discarded dataset: {len(df_discarded)} samples") 
    for class_val, count in discarded_class_counts.items():
        logging.info(f"  Class {class_val}: {count} samples")
    
    return df_balanced, df_discarded