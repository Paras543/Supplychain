import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.exception import CustomException
from src.logger import logging

import joblib


class TrainingPipeline:
    def __init__(self, source_path: str):
        self.source_path = source_path

    def run(self):
        logging.info("=== TRAINING PIPELINE STARTED ===")
        try:
            # Step 1: Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion(
                source_path=self.source_path
            )

            # Step 2: Transformation
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
                train_path, test_path
            )

            # Step 3: Training
            trainer = ModelTrainer()
            metrics = trainer.initiate_model_training(X_train, X_test, y_train, y_test)
            logging.info(f"Training metrics: {metrics}")

            # Step 4: Evaluation
            model = joblib.load(trainer.config.trained_model_path)
            evaluator = ModelEvaluation()
            report, cm_df = evaluator.initiate_model_evaluation(model, X_test, y_test)

            logging.info("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")

            return {
                'metrics': metrics,
                'classification_report': report,
                'confusion_matrix': cm_df,
            }

        except Exception as e:
            logging.error("Training pipeline failed")
            raise CustomException(e, sys)




