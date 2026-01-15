import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    try:
        logging.info("===== Training Pipeline Started =====")

        # 1️⃣ Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        # 2️⃣ Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr = transformation.initiate_data_transformation(
            train_path=train_path,
            test_path=test_path
        )
        logging.info("Data transformation completed")

        # 3️⃣ Model Training
        trainer = ModelTrainer()
        best_model_name, metrics = trainer.initiate_model_trainer(
            train_array=train_arr,
            test_array=test_arr
        )

        logging.info(f"Training completed. Best Model: {best_model_name}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
