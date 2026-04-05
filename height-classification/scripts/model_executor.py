import logging
import os
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, Any, List, Tuple
import pickle
import json
from datetime import datetime
import scipy.stats as stats

from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import (
    get_model_class,
    get_preprocessor_class,
    get_dim_reducer_class,
    get_cv_strategy_class,
    save_test_final_model_plot,
    apply_dataset_undersampling
)


class ModelExecutor:
    """Class for performing machine learning experiments.

    Attributes:
       config (Dict[str, Any]): configuration dictionary containing model and
                experiment settings
       model (Any): the machine learning model instance
       dr_config (Dict[str, Any], optional): dimensionality reduction configuration
       outer_cv_config (Dict[str, Any]): outer cross-validation strategy configuration
       inner_cv_config (Dict[str, Any]): inner cross-validation strategy configuration
       metrics (List[str]): list of evaluation metrics for model assessment
       refit_metric (str): primary metric used for model selection during grid search
       param_grid (Dict[str, Any]): hyperparameter grid for grid search
       preprocessing_steps (Dict[str, Any]): data preprocessing to apply
       experiment_name (str): name identifier for the current experiment
       result_directory (str): directory path for saving experiment results
       base_filename (str): filename to store the results for a given experiment
         based on experiment_name
       plot_file_path (str): path to the plot representing the test results of the
        final model
       undersampling (Dict[str, Any]): dictionary with parameters for undersampling
       results (Dict[str, Any]): dictionary storing all experiment results
       final_model (Any): the best trained model after hyperparameter optimization
       logger (logging.Logger): logger instance for tracking operations
    """

    def __init__(self, config: Dict[str, Any], results_directory):
        """Initialize the ModelExecutor with configuration and results directory path.

        Args:
            config (Dict[str, Any]): configuration dictionary
            results_directory (str): directory path where experiment results will be saved
        """

        self.results_directory = results_directory
        self.config = config

        # Define attributes from config
        self.model = get_model_class(config["model_class"])()

        self.dr_config = None
        if "dim_reduction" in config and config["dim_reduction"]:
            self.dr_config = config["dim_reduction"]

        self.outer_cv_config = config["outer_cross_validation"]
        self.inner_cv_config = config["inner_cross_validation"]
        self.metrics = config["metrics"]
        self.refit_metric = self.metrics[0]
        self.param_grid = config["parameters"]
        self.preprocessing_steps = config["preprocessing_steps"]
        self.experiment_name = config["experiment_name"]
        self.result_directory = results_directory
        self.base_filename = f"{self.experiment_name.lower().replace(' ', '_')}"
        self.plot_file_path = os.path.join(
            self.result_directory, "test_results_plots", f"{self.base_filename}.png"
        )
        self.undersampling=config.get("undersampling", False)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("DataExecutor initialized")
        self.logger.debug(f"Results directory: {results_directory}")

        self.results = {}
        self.final_model = None

    def test_train_data_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        stratify_column: str = None,
        test_size: float = 0.11,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training and test datasets using stratified sampling
        to maintain class distribution proportions in both splits.

        Args:
            - df (DataFrame): pandas DataFrame containing the complete dataset to split
            - target_column (str): name of the target column for classification
            - stratify_column (str, optional): column name to use for stratification.
                If None, uses target_column for stratification (default: None)
            - test_size (float): proportion of data to include in test set (default: 0.11)
            - random_state (int): random seed for reproducible results (default: 42)

        Returns:
            - df_train,df_test (tuple[DataFrame, DataFrame]):
                training and testing DataFrames (train, test)

        """

        self.logger.info("Performing train-test split")

        if stratify_column is None:
            stratify_data = df[target_column]
        else:
            stratify_data = df[stratify_column]

        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
            stratify=stratify_data,
        )

        return df_train, df_test

    def define_grid_search_estimator(self, pipeline: Pipeline) -> GridSearchCV:
        """
        Creates a GridSearchCV estimator for hyperparameter optimization using
        the configured inner cross-validation strategy.

        Args:
            - pipeline (Pipeline): scikit-learn pipeline containing preprocessing,
                 dimensionality reduction, and model steps

        Returns:
            - GridSearchCV: configured grid search estimator for hyperparameter
                optimization
        """

        self.logger.info("Defining grid search estimator")

        inner_cv_class = get_cv_strategy_class(self.inner_cv_config["strategy"])
        inner_cv_params = {
            k: v for k, v in self.inner_cv_config.items() if k != "strategy"
        }
        inner_cv = inner_cv_class(**inner_cv_params)

        self.logger.info(f"Inner CV defined: {inner_cv}")

        grid_search_estimator = GridSearchCV(
            pipeline,
            self.param_grid,
            cv=inner_cv,
            scoring=self.metrics,
            refit=self.refit_metric,
            n_jobs=1,
        )
        return grid_search_estimator

    def create_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> Pipeline:
        """
        Builds ML pipeline based on the configuration with preprocessing,
        dimensionality reduction, and model steps.

        Args:
            numerical_features (List[str]): list of numerical feature column names
            categorical_features (List[str]): list of categorical feature column names

        Returns:
            Pipeline: scikit-learn pipeline ready for training
        """
        self.logger.info("Creating Processing Pipeline")

        transformers = []

        for step in self.config["preprocessing_steps"]:
            preprocessor_class = get_preprocessor_class(step["name"])

            self.logger.info(f"Adding prepocessing step: {step['name']}")

            step_params = step.get("params", {})

            if step["apply_to"] == "numerical":
                self.logger.info(
                    f"Step applied to numerical features: {numerical_features}"
                )
                columns = numerical_features
            elif step["apply_to"] == "categorical":
                self.logger.info(
                    f"Step applied to categorical features: {categorical_features}"
                )
                columns = categorical_features
            else:
                columns = step["apply_to"]

            transformers.append(
                (
                    step["name"].lower(),
                    preprocessor_class(**step_params),
                    columns,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers, remainder="passthrough", n_jobs=1
        )

        pipeline_steps = [("preprocessing", preprocessor)]

        if self.dr_config:
            self.logger.info("Adding dimention reduction step in the pipeline")
            dr_class = get_dim_reducer_class(self.dr_config["name"])
            dim_reducer = dr_class()
            pipeline_steps.append(("dim_reduction", dim_reducer))

        self.logger.info("Adding model step in the pipeline")
        pipeline_steps.append(("model", self.model))

        return Pipeline(steps=pipeline_steps)

    def model_evaluation(
        self,
        grid_search_estimator: GridSearchCV,
        X_train: DataFrame,
        y_train: Series,
    ) -> Dict[str, float]:
        """Perform nested cross-validation to estimate model performance.

        Args:
            grid_search_estimator (GridSearchCV): configured grid search estimator
            X_train (DataFrame): training dataset containing features data
            y_train (Series): training target vector

        Returns:
            Dict[str, float]: dictionary containing performance metrics for each metric
             defined in the self.metrics with statistics:
                - scores_mean: average performance across CV folds
                - scores_std: standard deviation of performance
                - confidence_interval: 95% confidence interval statistics
        """

        self.logger.info("Starting model evaluation with nested cross validation")
        class_counts = y_train.value_counts()
        self.logger.info(
            f"Class distribution - Class 0: {class_counts.get(0, 0)}, "
            f"Class 1: {class_counts.get(1, 0)}"
        )

        outer_cv_class = get_cv_strategy_class(self.outer_cv_config["strategy"])
        outer_cv_params = {
            k: v for k, v in self.outer_cv_config.items() if k != "strategy"
        }
        outer_cv = outer_cv_class(**outer_cv_params)

        self.logger.info(f"Outer CV defined: {outer_cv}")

        nested_scores = {}
        for metric in self.metrics:
            self.logger.info(f"Performing Nested Cross Validation with Metric {metric}")
            scores = cross_val_score(
                grid_search_estimator,
                X_train,
                y_train,
                cv=outer_cv,
                scoring=metric,
                n_jobs=1,
            )

            confidence_level = 0.95
            degrees_freedom = len(scores) - 1
            sample_mean = scores.mean()
            sample_standard_error = stats.sem(scores)

            t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

            margin_of_error = t_critical * sample_standard_error
            ci_lower = sample_mean - margin_of_error
            ci_upper = sample_mean + margin_of_error

            nested_scores[metric] = {
                # 'scores': scores.tolist(),
                "scores_mean": scores.mean(),
                "scores_std": scores.std(),
                "confidence_interval": {
                    "confidence_level": confidence_level,
                    "lower_bound": ci_lower,
                    "upper_bound": ci_upper,
                    "margin_of_error": margin_of_error,
                    "standard_error": sample_standard_error,
                },
            }

        self.results["performance_estimation"] = {
            "description": "Performance estimated via nested cross-validation",
            "nested_cv": nested_scores,
        }

        return nested_scores

    def train_final_model(
        self,
        grid_search_estimator: GridSearchCV,
        X_train: DataFrame,
        y_train: Series,
    ) -> Any:
        """Fits the GridSearchCV estimator on the entire training dataset and
        stores the best estimator as the final model for predictions.

        Args:
            grid_search_estimator (GridSearchCV): configured grid search estimator
            X_train (DataFrame): training dataset containing features data
            y_train (Series): training target vector

        Returns:
            self.final_model (Any): the best trained model from grid search
        """

        self.logger.info(
            "Training the model to find the best estimator using the training data"
        )

        grid_search_estimator.fit(X_train, y_train)

        self.final_model = grid_search_estimator.best_estimator_
        self.results["final_model"] = {
            "description": "Best parameters found by fitting GridSearchCV "
            "on the entire dataset",
            "best_params": grid_search_estimator.best_params_,
            "best_score_metric": self.refit_metric,
            "best_score": grid_search_estimator.best_score_,
        }

        return self.final_model

    def save_final_model(self, grid_search_estimator: GridSearchCV) -> str:
        """Save the best trained model using pickle serialization.

        Args:
            grid_search_estimator (GridSearchCV): fitted grid search estimator

        Returns:
            best_model_path (str): path to the saved model .pkl file
        """

        self.logger.info("Saving the final model")

        filename = f"{self.base_filename}.pkl"
        models_directory = os.path.join(self.result_directory, "models")

        if models_directory and not os.path.exists(models_directory):
            os.makedirs(models_directory, exist_ok=True)

        best_model_path = os.path.join(models_directory, filename)

        with open(best_model_path, "wb") as f:
            pickle.dump(grid_search_estimator.best_estimator_, f)

        return best_model_path

    def test_final_model(
        self, X_test: DataFrame, y_test: DataFrame,
        blind_test: bool=False,
    ) -> tuple[DataFrame, DataFrame]:
        """Uses the trained final model on the test dataset to compute
        performance metrics (accuracy, f1_macro, precision_macro, recall_macro,
        precision_weighted, recall_weighted).

        Args:
            X_test (DataFrame): test dataframe with features data
            y_test (DataFrame): test target values

        Returns:
            y_test, y_pred (tuple[DataFrame, DataFrame]):
                actual test values and predicted values
        """
        if self.final_model is None:
            self.logger.error("No final model available. Train the model first.")
            raise ValueError("No final model available. Train the model first.")

        self.logger.info("Testing the final model")
        y_pred = self.final_model.predict(X_test)

        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }

        if blind_test:
            self.results["blind_test_performance"] = test_metrics
        else:
            self.results["test_performance"] = test_metrics

        return y_test, y_pred

    def save_results(self):
        """
        Save all results to a JSON file: performance metrics, model parameters,
        cross-validation results, and metadata.

        Args:
            filename (str): filename for the results JSON file
            dataset_name (str): name identifier for the dataset used
        """
        if self.result_directory and not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory, exist_ok=True)

        filename = f"{self.base_filename}.json"

        results_path = os.path.join(self.result_directory, filename)

        with open(results_path, "w") as f:
            json.dump(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "experiment_name": f"{self.experiment_name}",
                    "results": self.results,
                },
                f,
                indent=4,
            )

    def run_experiment(
        self,
        df: pd.DataFrame,
        column_class_name: str,
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> Dict[str, Any]:
        """
        Executes the complete machine learning experiment pipeline.

        Args:
            - df (DataFrame): complete dataset containing features and target variable
            - column_class_name (str): name of the target column for binary classification
            - numerical_features (List[str]): list of numerical feature column names
            - categorical_features (List[str]): list of categorical feature column names

        Returns:
            - Dict[str, Any]: comprehensive results dictionary containing:
                - performance_estimation: nested CV performance metrics with confidence intervals
                - final_model: best hyperparameters and training performance
                - test_performance: final model evaluation on test set including accuracy,
                f1_macro, precision, and recall metrics


        """

        self.logger.info("Start running the experiment")

        if self.undersampling:
            df, df_discarded = apply_dataset_undersampling(df, column_class_name, self.undersampling.get("strategy",'auto'))
        
        df_train, df_test = self.test_train_data_split(df, column_class_name)

        all_features = numerical_features + categorical_features
        X_train = df_train[all_features]
        y_train = df_train[column_class_name]
        X_test = df_test[all_features]
        y_test = df_test[column_class_name]

        self.logger.info(
            f"Training Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        self.logger.info(
            f"Test Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features"
        )

        pipeline = self.create_pipeline(numerical_features, categorical_features)
        grid_search_estimator = self.define_grid_search_estimator(pipeline)

        self.model_evaluation(grid_search_estimator, X_train, y_train)
        self.train_final_model(grid_search_estimator, X_train, y_train)
        self.save_final_model(grid_search_estimator)
        y_test, y_pred = self.test_final_model(X_test, y_test)
        save_test_final_model_plot(y_test, y_pred, self.plot_file_path)

        # Blind test
        if self.undersampling:

            X_discarded_test = df_discarded[all_features]
            y_discarded_test = df_discarded[column_class_name]

            y_test, y_pred = self.test_final_model(X_discarded_test, y_discarded_test, blind_test=True)

        self.save_results()

        return self.results
