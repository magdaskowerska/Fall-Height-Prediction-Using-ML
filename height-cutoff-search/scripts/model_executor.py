from pandas import DataFrame, Series
from typing import List, Dict, Any
import json
import pickle
import os
from datetime import datetime
import logging
import scipy.stats as stats
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,confusion_matrix,accuracy_score, f1_score

from utils import (
    get_model_class,
    get_preprocessor_class,
    get_dim_reducer_class,
    get_cv_strategy_class,
    save_test_final_model_plot,
    save_test_performance_binary_var_plot
)


class ModelExecutor:
    """Class for performing machine learning experiments.

    Attributes:
       config (Dict[str, Any]): configuration dictionary containing model and
                experiment settings
       model (Any): the machine learning model instance
       dr_config (Dict[str, Any] | None): dimensionality reduction configuration
       outer_cv_config (Dict[str, Any]): outer cross-validation strategy configuration
       inner_cv_config (Dict[str, Any]): inner cross-validation strategy configuration
       metrics (List[str]): list of evaluation metrics for model assessment
       refit_metric (str): primary metric used for model selection during grid search
       param_grid (Dict[str, Any]): hyperparameter grid for grid search
       dual_evaluation_at_height_cutoff (int| None): integer used for the dual evaluation of the model
       experiment_name (str): name identifier for the current experiment
       result_directory (str): directory path for saving experiment results
       models_directory (str): directory path for saving trained models
       results (Dict[str, Any]): dictionary storing all experiment results
       final_model (Any): the best trained model after hyperparameter optimization
       logger (logging.Logger): logger instance for tracking operations
    """

    def __init__(
        self,
        config: Dict[str, Any],
        results_directory: str,
        models_directory: str,
    ):
        """Initialize the ModelExecutor with configuration and directory paths.

        Args:
            config (Dict[str, Any]): configuration dictionary
            results_directory (str): directory path where experiment results will be saved
            models_directory (str): directory path where trained models will be saved
        """

        self.config = config

        self.model = get_model_class(config["model_class"])()

        self.dr_config = None
        if "dim_reduction" in config and config["dim_reduction"]:
            self.dr_config = config["dim_reduction"]

        self.outer_cv_config = config["outer_cross_validation"]
        self.inner_cv_config = config["inner_cross_validation"]
        self.metrics = config.get("metrics", ["neg_mean_absolute_error"])
        self.refit_metric = config.get("refit_metric", self.metrics[0])
        self.param_grid = config["parameters"]
        self.experiment_name = config["experiment_name"]
        self.dual_evaluation_at_height_cutoff=config.get("dual_evaluation_at_height_cutoff", None)
        self.result_directory = results_directory
        self.models_directory = models_directory

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("DataExecutor initialized")
        self.logger.debug(f"Results directory: {results_directory}")
        self.logger.debug(f"Models will be saved in the directory: {models_directory}")

        self.results = {}
        self.final_model = None

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

    def test_final_model(
        self, X_test: DataFrame, y_test: Series
    ) -> tuple[Series, Series]:
        """Uses the trained final model on the test dataset to compute
        performance metrics. Depending on configuration, computes either
        standard regression metrics (MAE, MSE, R²) or dual evaluation
        metrics (classification above cutoff + regression below cutoff).

        Args:
            X_test (DataFrame): test dataframe with features data
            y_test (Series): test target values

        Returns:
            y_test, y_pred (tuple[Series, Series]):
                actual test values and predicted values
        """
        if self.final_model is None:
            self.logger.error("No final model available. Train the model first.")
            raise ValueError("No final model available. Train the model first.")

        self.logger.info("Testing the final model")
        y_pred = Series(self.final_model.predict(X_test), index=y_test.index) 

        if height_cutoff := self.dual_evaluation_at_height_cutoff:
            test_metrics = self._compute_dual_metrics(y_test, y_pred, height_cutoff)
        else:
            test_metrics = self._compute_regression_metrics(y_test, y_pred)

        self.results["test_performance"] = test_metrics
        return y_test, y_pred
    
    def _compute_regression_metrics(self, y_true: Series, y_pred: Series) -> Dict[str, float]:
        """Compute standard regression metrics.
        Args:
            y_true (Series): true target values
            y_pred (Series): predicted target values

        Returns:
            Dict[str, float]: dictionary containing the following metrics:
                - mae: mean absolute error
                - mse: mean squared error
                - r2: R² score
        """
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

    def _compute_dual_metrics(
        self, y_test: Series, y_pred: Series, height_cutoff: int
    ) -> Dict[str, Any]:
        """Compute dual evaluation metrics: binary classification above cutoff
        and regression below cutoff.
        
        Args:
            y_test (Series): true target values
            y_pred (Series): predicted target values
            height_cutoff (int): integer used for the dual evaluation of the model

        Returns:
            Dict[str, Any]: dictionary with two keys:
                - higher_than_cutoff: accuracy, f1_macro and confusion matrix values
                - lower_than_cutoff: mae, mse, r2
        """
        higher_mask = y_test > height_cutoff
        lower_mask = ~higher_mask 

        y_test_higher = y_test[higher_mask]
        y_pred_higher = y_pred[higher_mask]
        y_test_higher_binary = np.ones(len(y_test_higher), dtype=int)
        y_pred_higher_binary = (y_pred_higher >= height_cutoff).astype(int)

        cm = confusion_matrix(y_test_higher_binary, y_pred_higher_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        return {
            "higher_than_cutoff": {
                "accuracy": accuracy_score(y_test_higher_binary, y_pred_higher_binary),
                "f1_macro": f1_score(y_test_higher_binary, y_pred_higher_binary, average="macro", zero_division=0),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "lower_than_cutoff": self._compute_regression_metrics(
                y_test[lower_mask], y_pred[lower_mask]
            ),
        }

    def identify_and_save_predicted_outliers(
        self,
        X_test: DataFrame,
        y_test: Series,
        y_pred: Series,
        outliers_data_path: str,
        threshold_std: float = 2.0,
    ):
        """Identifies test samples where the prediction residuals exceed a
        threshold based on standard deviations.

        Args:
            X_test (DataFrame): test dataframe with features data
            y_test (Series): test target values
            y_pred (Series): predicted target values
            outliers_data_path (str): CSV file path to save identified outliers
            threshold_std (float): standard deviation multiplier for
                    outlier threshold (default: 2.0)
        """
        residuals = y_test - y_pred
        absolute_residuals = np.abs(residuals)

        residual_std = residuals.std()
        threshold = threshold_std * residual_std

        outlier_mask = absolute_residuals > threshold

        if outlier_mask.sum() == 0:
            self.logger.info("no outliers identified in the predicted values")
            return

        outliers_df = X_test[outlier_mask].copy()
        outliers_df["y_test"] = y_test[outlier_mask]
        outliers_df["y_pred"] = y_pred[outlier_mask]
        outliers_df["residual"] = residuals[outlier_mask]

        if not os.path.exists(os.path.dirname(outliers_data_path)):
            os.makedirs(os.path.dirname(outliers_data_path), exist_ok=True)

        outliers_df.to_csv(outliers_data_path, index=False)


    def create_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> Pipeline:
        """Builds ML pipeline based on the configuration with preprocessing,
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

    def save_final_model(
        self, grid_search_estimator: GridSearchCV, filename: str
    ) -> str:
        """Save the best trained model using pickle serialization.

        Args:
            grid_search_estimator (GridSearchCV): fitted grid search estimator
            filename (str): filename for saving the model

        Returns:
            best_model_path (str): path to the saved model .pkl file
        """

        self.logger.info("Saving the final model")

        if self.models_directory and not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory, exist_ok=True)

        best_model_path = os.path.join(self.models_directory, filename)

        with open(best_model_path, "wb") as f:
            pickle.dump(grid_search_estimator.best_estimator_, f)

        return best_model_path

    def save_results(self, filename: str, dataset_name: str):
        """
        Save all results to a JSON file: performance metrics, model parameters,
        cross-validation results, and metadata.

        Args:
            filename (str): filename for the results JSON file
            dataset_name (str): name identifier for the dataset used
        """
        if self.result_directory and not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory, exist_ok=True)

        results_path = os.path.join(self.result_directory, filename)

        with open(results_path, "w") as f:
            json.dump(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "experiment_name": f"{self.experiment_name} on {dataset_name}",
                    "results": self.results,
                },
                f,
                indent=4,
            )

    def _save_plots(self, y_test: Series, y_pred: Series, plot_path: str):
        """Save appropriate plots depending on whether dual evaluation is configured.
        Args:
            y_test (Series): true target values
            y_pred (Series): predicted target values
            plot_path (str): file path for saving the plot
        """
        if height_cutoff := self.dual_evaluation_at_height_cutoff:
            lower_mask = y_test <= height_cutoff
            save_test_final_model_plot(y_test[lower_mask], y_pred[lower_mask], plot_path)

            perf = self.results["test_performance"]["higher_than_cutoff"]
            binary_plot_path = os.path.join(
                self.result_directory,
                "plot_result_heights_exceed_cutoff",
                os.path.basename(plot_path),
            )
            save_test_performance_binary_var_plot(
                perf["true_positives"], perf["true_negatives"],
                perf["false_positives"], perf["false_negatives"],
                binary_plot_path,
            )
        else:
            save_test_final_model_plot(y_test, y_pred, plot_path)

    def run_experiment(
        self,
        df_train,
        df_test,
        numerical_features: List[str],
        categorical_features: List[str],
        dataset_name: str,
        y_column: str = "height_of_fall_m",
    ):
        """Execute the complete machine learning experiment pipeline.

        Args:
            df_train (DataFrame): training dataset
            df_test (DataFrame): test dataset
            numerical_features (List[str]): names of numerical feature columns
            categorical_features (List[str]): names of categorical feature columns
            dataset_name (str): identifier for the dataset being used
            y_column (str): name of the target column (default: 'height_of_fall_m')

        Returns:
            Dict[str, Any]: complete results dictionary containing:
                - performance_estimation: nested CV results with confidence intervals
                - final_model: best model parameters and training performance
                - test_performance: final model performance on test set
        """

        self.logger.info("Starting running the experiment")
        all_features = numerical_features + categorical_features
        X_train = df_train[all_features]
        y_train = df_train[y_column]
        X_test=df_test[all_features]
        y_test = df_test[y_column]

        self.logger.info(
            f"Training Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        self.logger.info(
            f"Test Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features"
        )

        inner_cv_class = get_cv_strategy_class(self.inner_cv_config["strategy"])
        inner_cv_params = {
            k: v for k, v in self.inner_cv_config.items() if k != "strategy"
        }
        inner_cv = inner_cv_class(**inner_cv_params)

        self.logger.info(f"Inner CV defined: {inner_cv}")

        # define the pipeline
        pipeline = self.create_pipeline(numerical_features, categorical_features)

        self.logger.info("Defining grid search estimator")
        grid_search_estimator = GridSearchCV(
            pipeline,
            self.param_grid,
            cv=inner_cv,
            scoring=self.metrics,
            refit=self.refit_metric,
            n_jobs=1,
        )

        # Define filename for storing the results and the model
        base_filename = (
            f"{self.experiment_name.lower().replace(' ', '_')}_{dataset_name}"
        )
        results_filename = f"{base_filename}.json"
        model_filename = f"{base_filename}.pkl"
        plot_test_final_model_filename = f"{base_filename}.png"
        outliers_data_filename = f"{base_filename}.csv"

        plot_test_final_model_path = os.path.join(
            self.result_directory,
            "plots_pred_vs_real_y",
            plot_test_final_model_filename,
        )

        outliers_data_path = os.path.join(
            self.result_directory,
            "outliers_predicted_y",
            outliers_data_filename,
        )

        self.model_evaluation(grid_search_estimator, X_train, y_train)
        self.train_final_model(grid_search_estimator, X_train, y_train)
        self.save_final_model(grid_search_estimator, model_filename)
        y_test, y_pred = self.test_final_model(X_test, y_test)
        self.save_results(results_filename, dataset_name)
        self.identify_and_save_predicted_outliers(
            X_test, y_test, y_pred, outliers_data_path=outliers_data_path
        )

        self._save_plots(y_test, y_pred, plot_test_final_model_path) 

        return self.results
