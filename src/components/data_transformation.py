import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # """Create domain-driven features"""

        logging.info("Starting feature engineering")

        # Temperature difference
        df["Temp_diff"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

        # Overheat risk
        df["Overheat_Risk"] = (df["Temp_diff"] > 20).astype(int)

        # Mechanical power
        df["Power"] = (
            df["Torque [Nm]"] * df["Rotational speed [rpm]"]
        )

        # High load flag
        df["High_Load"] = (
            df["Torque [Nm]"] >
            df["Torque [Nm]"].quantile(0.75)
        ).astype(int)

        # Tool wear features
        df["Tool_Wear_Ratio"] = df["Tool wear [min]"] / 240
        df["Tool_Expiry_Risk"] = (
            df["Tool_Wear_Ratio"] > 0.8
        ).astype(int)

        # Mechanical Stress Index
        df["Mechanical_Stress_Index"] = (
            (df["Torque [Nm]"] / df["Torque [Nm]"].max()) * 0.4 +
            (df["Temp_diff"] / df["Temp_diff"].max()) * 0.3 +
            df["Tool_Wear_Ratio"] * 0.3
        )

        logging.info("Feature engineering completed")

        return df

    def get_data_transformer_object(self):
        try:
            logging.info("Creating preprocessing pipeline")

            categorical_cols = ["Product ID", "Type"]
            numerical_cols = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Temp_diff",
                "Overheat_Risk",
                "Power",
                "High_Load",
                "Tool_Wear_Ratio",
                "Tool_Expiry_Risk",
                "Mechanical_Stress_Index"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str
    ):

        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "Machine failure"

            # Leakage columns
            leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

            X_train = train_df.drop(columns=[target_column] + leakage_cols)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column] + leakage_cols)
            y_test = test_df[target_column]

            # Feature engineering
            X_train = self._feature_engineering(X_train)
            X_test = self._feature_engineering(X_test)

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            # If output is sparse, convert to dense
            if hasattr(X_train_transformed, "toarray"):
                X_train_transformed = X_train_transformed.toarray()
            if hasattr(X_test_transformed, "toarray"):
                X_test_transformed = X_test_transformed.toarray()
            # Save preprocessor
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                exist_ok=True
            )

            
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            train_arr = np.c_[X_train_transformed, y_train.values.reshape(-1, 1)]
            test_arr = np.c_[X_test_transformed, y_test.values.reshape(-1, 1)]

            logging.info("Preprocessor saved successfully")

            logging.info("Data transformation completed")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)
