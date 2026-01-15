import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(y_true, y_pred):
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }

        logging.info(f"Model evaluation metrics: {metrics}")
        return metrics

    except Exception as e:
        raise CustomException(e, sys)
