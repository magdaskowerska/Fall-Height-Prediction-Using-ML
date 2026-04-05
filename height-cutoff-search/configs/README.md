# Experiment Configuration Documentation

## .JSON Schema Description for Data Processing and Preparation

- `experiment_name`: experiment description
- `cap_column_values` (Optional): json for capping or removing values greater than the cutoff value
    - `column_name`: string representing the numeric column
    - `cutoff_value`: integer representing the maximum value
    - `method` (Optional): can be 'cap' (default value) if we want the values to be capped or 'remove' if we want to remove the records above the cutoff_value
- `normalization` (Optional): normalize the 'height_of_fall_m' column to the nearest multiple of 3
- `remove_outliers` (Optional): json for removing outliers of a column
    - `threshold` (Optional): float representing the IQR multiplier for outlier detection. Common values: 1.5 (standard), 2.0 (less strict), 3.0 (very lenient). 
    - `column_name`: numeric column for removing outliers

## Example:
```json
{
    "experiment_name": "Example experiment",
    "cap_column_values": {
        "column_name": "height_of_fall_m",
        "cutoff_value": 24,
        "method": "cap"
    },
    "remove_outliers": {
        "threshold": 3.0,
        "column_name": "height_of_fall_m"
    },
    "normalization": "True"
}
```

## .JSON Schema Description for ML Models Configuration


- `experiment_name`: descriptive name for the machine learning experiment
- `model_class`: string name of the ML model class to use (see supported models below)
- `parameters`: dictionary containing hyperparameter grid for grid search optimization
- `metrics`: list of evaluation metrics for model assessment
- `refit_metric`: primary metric used for model selection during grid search (must be in metrics list)
- `features` (Optional): used to specify different feature sets:
    - `'all'`: all numerical features in the dataset
    - `'no_bin'`: features without binary variables
    - `'no_bin_no_agg'`: only core demographic features
    - `'bin'`: all numerical except anatomical region features
- `preprocessing_steps`: list of preprocessing configurations to apply
    - `name`: string name of the preprocessing class (see supported preprocessing methods below)
    - `apply_to`: target feature type ('numerical', 'categorical', or 'all')
    - `params` (Optional): dictionary of parameters for the preprocessing step
- `outer_cross_validation`: configuration for outer CV loop (for performance estimation) (see supported cross-validation techniques below)
    - `strategy`: CV strategy name ('KFold', 'StratifiedKFold', 'TimeSeriesSplit')
    - `n_splits`: number of CV folds
    - Additional parameters for CV like: `shuffle`, `random_state`.
- `inner_cross_validation`: configuration for inner CV loop (for hyperparameter tuning)
    - Same structure as outer_cross_validation
- `dim_reduction` (Optional): dimensionality reduction configuration (see supported dimentionality reduction below)
    - `method`: reduction technique name (example: 'PCA', 'SVD', 'ICA')
    - `params`: parameters for the dimensionality reduction method
- `dual_evaluation_at_height_cutoff` (Optional): integer representing a secondary height cutoff value in meters used for the dual evaluation of the model with heights higher than the cutoff value evaluated with the standard classification metrics while heights below the cutoff value with MAE

## Example:
```json
{
        "experiment_name": "Kernel Ridge RobustScaler SVD",
        "model_class": "KernelRidge",
        "enable": true,
        "parameters": {
            "dim_reduction__n_components": [15, 20, 25],
            "model__alpha":  [0.01, 0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear", "poly"],
            "model__degree":[2, 3, 4],
            "model__gamma": [0.01, 0.1, 1.0]
        },
        "features": "bin",
        "preprocessing_steps": [
            {"name": "OrdinalEncoder", 
            "apply_to": "categorical",
            "params": {
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
            "encoded_missing_value": -1
        }},
            {"name": "RobustScaler", "apply_to": "numerical"}
        ],
        "outer_cross_validation": {
            "strategy": "KFold",
            "n_splits": 5,
            "shuffle": true,
            "random_state": 24
        },
        "inner_cross_validation": {
            "strategy": "KFold",
            "n_splits": 3,
            "shuffle": true,
            "random_state": 24
        },
        "metrics": ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        "dim_reduction": {
            "name": "SVD"
        }
    }
```

### Supported Components

**Models:**
- `LinearRegression`: Linear regression model
- `BayesianRidge`: Bayesian ridge regression
- `SVR`: Support Vector Regression
- `KernelRidge`: Kernel ridge regression
- `DecisionTreeRegressor`: Decision tree regressor
- `RandomForestRegressor`: Random forest regressor

**Preprocessing:**
- `StandardScaler`: Standardize features by removing mean and scaling to unit variance
- `RobustScaler`: Scale features using statistics robust to outliers
- `MinMaxScaler`: Scale features to a given range (default 0-1)
- `OrdinalEncoder`: Encode categorical features as ordinal integers

**Dimensionality Reduction:**
- `PCA`: Principal Component Analysis
- `SVD`: Truncated Singular Value Decomposition
- `ICA`: Independent Component Analysis

**Cross-Validation Strategies:**
- `KFold`: K-Fold cross-validation
- `StratifiedKFold`: Stratified K-Fold cross-validation
- `TimeSeriesSplit`: Time series cross-validation