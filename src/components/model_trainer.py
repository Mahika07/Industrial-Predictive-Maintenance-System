import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts", "model.pkl"
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42
                ),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
                    random_state=42
                )
            }

            model_report = {}

            for model_name, model in models.items():
                metrics = self.evaluate_model(
                    model, X_train, y_train, X_test, y_test
                )
                model_report[model_name] = metrics

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["roc_auc"]
            )
            best_model = models[best_model_name]

            best_model.fit(X_train, y_train)

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            logging.info(f"Best model selected: {best_model_name}")
            logging.info(f"Model performance: {model_report[best_model_name]}")

            return best_model_name, model_report

        except Exception as e:
            raise CustomException(e, sys)




