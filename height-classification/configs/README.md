# Experiment Configuration Documentation

## .JSON Schema Description for Data specification

- `experiment_name`: dataset description
- `filename_data`: the .csv file name in the data/ directory containing the dataset
- `height_classification_threshold`: height threshold in meters ex.: 24 to build the target binary variable. Values equal to 1 will be given to heights greater than 24, otherwise the variable is set to 0.
- `features` (Optional): used to specify different feature sets:
    - `'all'`: all numerical features in the dataset
    - `'no_bin'`: features without binary variables
    - `'no_bin_no_agg'`: only core demographic features
    - `'bin'`: all numerical except anatomical region features


## Example:
```json
{
    "experiment_name": "Example experiment",
    "filename_data":"data_example.csv",
    "height_classification_threshold": 24,
    "features": "bin" 
}
```

## .JSON Schema Description for ML Models Configuration

- `experiment_name`: descriptive name for the machine learning experiment
- `model_class`: string name of the ML model class to use (see supported models below)
- `parameters`: dictionary containing hyperparameter grid for grid search optimization
- `metrics`: list of evaluation metrics for model assessment, `f1_macro` and `accuracy` are mandatory
- `refit_metric` (Optional): primary metric used for model selection during grid search (must be in metrics list)
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
- `undersampling` (Optional): configuration for random undersampling of the majority class to balance the dataset
    - `strategy`: sampling strategy to use (default: `'custom'`). It can be `custom` or one of the strategies supported by the RandomUnderSampler class from the python imbalanced-learn library
    - `random_state`: random seed for reproducibility (default: 42)
    - `preserve_ratio` (float)(Optional): fraction of minority class to remove from training and add to discarded dataset for blind testing (default: 0.1)


## Example:
```json
{
        "experiment_name": "Logistic Regression Classification",
        "model_class": "LogisticRegression",
        "enable": true,
        "parameters": {
            "dim_reduction__n_components": [ 10, 15, 20, 25],
            "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": [
                "liblinear",
                "saga"
            ],
            "model__penalty": [
                "l1",
                "l2"
            ],
            "model__max_iter": [1000]
        },
        "preprocessing_steps": [
            {
                "name": "OrdinalEncoder",
                "apply_to": "categorical",
                "params": {
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1,
                    "encoded_missing_value": -1
                }
            },
            {
                "name": "RobustScaler",
                "apply_to": "numerical"
            }
        ],
        "outer_cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": true,
            "random_state": 42
        },
        "inner_cross_validation": {
            "strategy": "StratifiedKFold",
            "n_splits": 3,
            "shuffle": true,
            "random_state": 42
        },
        "metrics": [
            "accuracy",
            "f1_macro"
        ],
        "dim_reduction": {
            "name": "SVD"
        }
    }
```

### Supported Components

**Models:**
- `LogisticRegression`: Logistic regression for binary classification
- `SVC`: Support Vector Classifier for classification tasks
- `GaussianNB`: Gaussian Naive Bayes classifier
- `KNeighborsClassifier`: K-Nearest Neighbors classifier
- `XGBClassifier`: XGBoost gradient boosting classifier
- `RandomForestClassifier`: Random forest ensemble classifier

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