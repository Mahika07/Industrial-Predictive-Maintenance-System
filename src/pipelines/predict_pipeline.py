import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["Temp_diff"] = (
                df["Process temperature [K]"] - df["Air temperature [K]"]
            )

            df["Overheat_Risk"] = (df["Temp_diff"] > 20).astype(int)

            df["Power"] = (
                df["Torque [Nm]"] * df["Rotational speed [rpm]"]
            )

            df["High_Load"] = (
                df["Torque [Nm]"] >
                df["Torque [Nm]"].quantile(0.75)
            ).astype(int)

            df["Tool_Wear_Ratio"] = df["Tool wear [min]"] / 240

            df["Tool_Expiry_Risk"] = (
                df["Tool_Wear_Ratio"] > 0.8
            ).astype(int)

            df["Mechanical_Stress_Index"] = (
                (df["Torque [Nm]"] / df["Torque [Nm]"].max()) * 0.4 +
                (df["Temp_diff"] / df["Temp_diff"].max()) * 0.3 +
                df["Tool_Wear_Ratio"] * 0.3
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, input_data: dict):
        try:
            # Convert dict to DataFrame
            df = pd.DataFrame([input_data])

            logging.info("Input data converted to DataFrame")
            df = self._feature_engineering(df)

          # Transform features
            X_transformed = self.preprocessor.transform(df)

            # Prediction
            prediction = self.model.predict(X_transformed)[0]

            # Probability (VERY IMPORTANT FIX)
            probability = self.model.predict_proba(X_transformed)[0][1]

            return {
                "prediction": int(prediction),
                "failure_probability": round(float(probability), 2)
            }

        except Exception as e:
            raise CustomException(e, sys)