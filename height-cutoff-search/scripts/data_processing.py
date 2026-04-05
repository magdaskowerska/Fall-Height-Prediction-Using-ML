import pandas as pd
from pandas import DataFrame
from typing import Optional, Literal
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging


class DataProcessor:
    """Data Processing class for handling CSV file and applying transformations
    for Machine Learning experiments.

    Attributes:
        path_to_csv_file (str): Path to the input CSV file
        dir_processed_data (str): Directory path where processed data will be saved
        logger (logging.Logger): Logger instance for tracking operations
    """

    def __init__(self, path_to_csv_file: str, dir_processed_data: str):
        """Initialize the Data Processor with input and output paths to data.

        Args:
            path_to_csv_file (str): path to the input CSV file
                with data after initial cleaning
            dir_processed_data (str): directory path where processed
                data will be saved
        """
        self.path_to_csv_file = path_to_csv_file
        self.dir_processed_data = dir_processed_data

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("DataProcessor initialized")
        self.logger.debug(f"Input file: {path_to_csv_file}")
        self.logger.debug(f"Output directory: {dir_processed_data}")

    def load_data_from_csv(self, filename: Optional[str] = None) -> DataFrame:
        """Load the dataset from CSV file into a pandas DataFrame.

        Args:
            filename (Optional[str]): path to the csv file to be load. If None,
                it loads the file at the path_to_csv_file path.
        Returns:
            df (DataFrame): pandas dataframe with the data loaded from the CSV file.
        """
        if filename:
            file_path = filename
        else:
            file_path = self.path_to_csv_file

        self.logger.info(f"Loading data from: {file_path}")

        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully from: {file_path}")
            return df
        except FileNotFoundError as exc:
            self.logger.error(f"CSV file not found at path: {file_path}")
            raise FileNotFoundError(f"CSV file not found at path: {file_path}") from exc
        except pd.errors.ParserError as exc:
            self.logger.error(f"Failed to parse CSV file: {file_path}")
            raise ValueError(f"Failed to parse CSV file: {file_path}") from exc

    def cap_values(
        self,
        df: DataFrame,
        cutoff_value: int,
        column_name: str,
        method: Optional[Literal["cap", "remove"]] = "cap",
    ) -> DataFrame:
        """Cap or remove values greater than a specified threshold in the
        DataFrame column in input.

        Args:
            df (DataFrame): DataFrame to process
            cutoff_value (int): threshold to use for capping or removing data
            column_name(str): name of the column for applying the processing
            method(Optional[Literal['cap','remove']]): processing method.
                The values higher than the threshold are replaced by the
                threshold in case of 'cap' method or removed in case of 'remove' method.
                With the method 'remove' the entire record will be removed.
                The default method is 'cap'.

        Returns:
            df_capped (DataFrame): processed pandas DataFrame.
        """

        self.logger.info(
            f"Applying {method} method with cutoff {cutoff_value} to column '{column_name}'"
        )

        df_capped = df.copy()
        if column_name not in df_capped.columns:
            self.logger.error(f"Column {column_name} not in the dataframe")
            raise ValueError(f"Column {column_name} not in the dataframe")

        if not pd.api.types.is_numeric_dtype(df_capped[column_name]):
            self.logger.error(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be capped."
            )
            raise ValueError(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be capped."
            )

        if method == "cap":
            df_capped[column_name] = np.where(
                df_capped[column_name] > cutoff_value,
                cutoff_value,
                df_capped[column_name],
            )
            self.logger.info(f"Values greater than {cutoff_value} were capped")

        elif method == "remove":
            df_capped = df_capped[df_capped[column_name] <= cutoff_value]
            self.logger.info(f"Values greater than {cutoff_value} were removed")

        else:
            raise ValueError("Select one of the following methods: remove, cap")

        return df_capped

    def normalize_height_values(
        self, df: DataFrame, column_name: str = "height_of_fall_m"
    ) -> DataFrame:
        """Normalize height values by rounding it to the neareast value among
        list of values representing floor levels starting from 6 and increasing
        in 3-meter increments (6, 9, 12, 15, ...).

        Args:
            df (DataFrame): pandas dataframe in input
            column_name (str): name of the column to normalize,
                with the default 'height_of_fall_m'

        Returns:
            df_normalized (DataFrame): dataframe with the normalized column

        """
        df_normalized = df.copy()
        max_height = int(max(df_normalized[column_name]))
        self.logger.info(
            f"Normalizing column '{column_name}' to floor values "
            f"with max height {max_height}m"
        )

        if column_name not in df_normalized.columns:
            self.logger.error(f"Column {column_name} not in the dataframe")
            raise ValueError(f"Column {column_name} not in the dataframe")

        if not pd.api.types.is_numeric_dtype(df_normalized[column_name]):
            self.logger.error(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be normalize."
            )
            raise ValueError(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be normalize."
            )

        bins = list(range(6, max_height + 1, 3))

        df_normalized[column_name] = df_normalized[column_name].apply(
            lambda x: min(bins, key=lambda h: abs(h - x))
        )

        self.logger.info("Normalization ended successfully.")

        return df_normalized

    def remove_outliers(
        self, df: DataFrame, column_name: str, threshold: float = 1.5
    ) -> DataFrame:
        """Remove records wtih outliers in a given column.

        Args:
            df (DataFrame): pandas dataframe in input
            column_name (str): name of the column to normalize
            threshold (floor): IQR multiplier for outlier detection (default: 1.5)

        Returns:
            df_removed_outliers (DataFrame): output dataframe with removed outliers
        """

        self.logger.info(
            f"Removing outliers from column '{column_name}' "
            f"with IQR threshold {threshold}"
        )

        df_removed_outliers = df.copy()
        if column_name not in df_removed_outliers.columns:
            self.logger.error(f"Column {column_name} not in the dataframe")
            raise ValueError(f"Column {column_name} not in the dataframe")

        if not pd.api.types.is_numeric_dtype(df_removed_outliers[column_name]):
            self.logger.error(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be processed for outliers."
            )
            raise ValueError(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be processed for outliers."
            )

        Q1 = df_removed_outliers[column_name].quantile(0.25)
        Q3 = df_removed_outliers[column_name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        df_removed_outliers = df_removed_outliers[
            (df_removed_outliers[column_name] >= lower_bound)
            & (df_removed_outliers[column_name] <= upper_bound)
        ]

        removed_rows_count = len(df) - len(df_removed_outliers)

        self.logger.info(
            f"Removed {removed_rows_count} outliers, the number of records remained: {len(df_removed_outliers)}"
        )

        return df_removed_outliers

    def save_data_to_csv(self, df: DataFrame, processing_steps: list[str]):
        """Save processed DataFrame to a CSV file in the dir_processed_data
        directory, with the filename generated based on the processing steps in
        input.

        Args:
            df (DataFrame): DataFrame to save
            processing_steps (list[str]): List of processing step names for filename generation

        """

        self.logger.info("Saving processed data")

        if df is None or df.empty:
            self.logger.error("Cannot save empty DataFrame")
            raise ValueError("Cannot save empty DataFrame")

        if self.dir_processed_data and not os.path.exists(self.dir_processed_data):
            os.makedirs(self.dir_processed_data, exist_ok=True)

        steps_str = "_".join(processing_steps) if processing_steps else ""
        filename = f"berlin_{steps_str}.csv"

        path = os.path.join(self.dir_processed_data, filename)

        try:
            df.to_csv(path, index=False)
            self.logger.info(f"Data saved successfully at path: {path}")
        except Exception as e:
            self.logger.error(f"The file cannot be save at path {path}: {e}")
            raise ValueError(f"The file cannot be save at path {path}: {e}")

    def test_train_data_split(
        self,
        df: DataFrame,
        test_size: float = 0.11,
        column_name: str = "height_of_fall_m",
        random_state: int = 42,
    ) -> tuple[DataFrame, DataFrame]:
        """Split the DataFrame into training and testing datasets. First, it
        attempts to stratify the data by creating 4 quartiles on the specified
        column. If not enough values are rpesent a random split is performed.

        Args:
            df (DataFrame): input DataFrame to split
            test_size (float): proportion of data to include in test set (default: 0.11)
            column_name (str): column name for stratification (default: 'height_of_fall_m')
            random_state (int): random seed for reproducible results (default: 42)

        Returns:
            df_train,df_test (tuple[DataFrame, DataFrame]):
                training and testing DataFrames (train, test)

        """

        self.logger.info(
            f"Starting train-test split with test_size={test_size}, "
            f"stratified by '{column_name}'"
        )

        if df is None or df.empty:
            self.logger.error("Cannot split empty DataFrame")
            raise ValueError("Cannot split empty DataFrame")

        if column_name not in df.columns:
            self.logger.error(f"Column '{column_name}' not in the dataframe")
            raise ValueError(f"Column '{column_name}' not in the dataframe")

        if not pd.api.types.is_numeric_dtype(df[column_name]):
            self.logger.error(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be processed for outliers."
            )
            raise ValueError(
                f"Column {column_name} is not numeric. "
                "Only numeric columns can be processed for outliers."
            )

        df_to_stratify = df.copy()
        value_counts = df_to_stratify[column_name].value_counts()
        min_number_of_records_per_height = value_counts.min()

        bins = pd.qcut(
            df_to_stratify[column_name], q=4, retbins=True, duplicates="drop"
        )[1]

        if min_number_of_records_per_height < 2:
            self.logger.info("Some heights have less than 2 samples.")
            self.logger.info(
                "A simple random split without stratification must be performed."
            )

            stratify_column = None

        elif len(set(bins)) <= 4:
            self.logger.info(
                "Cannot create 4 quartiles for stratification "
                f"- bin edges less than 5 ({bins})."
            )
            self.logger.info(
                f"Using the column {column_name} values for stratification"
            )
            stratify_column = df[column_name]
        else:
            df_to_stratify.loc[:, "quartile"] = pd.cut(
                df_to_stratify[column_name],
                bins=bins,
                include_lowest=True,
                right=True,
            )
            stratify_column = df_to_stratify["quartile"]

        self.logger.info("Performing train-test split")
        df_train, df_test = train_test_split(
            df_to_stratify,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
            stratify=stratify_column,
        )

        if "quartile" in df_train.columns:
            df_train = df_train.drop(columns=["quartile"])
        if "quartile" in df_test.columns:
            df_test = df_test.drop(columns=["quartile"])

        self.logger.info("Train-test split completed successfully")

        return df_train, df_test
