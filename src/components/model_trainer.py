import os
import sys
from dataclasses import dataclass

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, accuracy_score
)
import joblib

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'xgb_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        logging.info("Model training started")
        try:
            model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)

            logging.info("Fitting XGBoost on training data")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'pr_auc': average_precision_score(y_test, y_proba),
            }
            logging.info(f"Test metrics: {metrics}")

            
            y_train_pred = model.predict(X_train)
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = metrics['f1']
            gap = train_f1 - test_f1
            logging.info(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Gap: {gap:.4f}")

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            joblib.dump(model, self.config.trained_model_path)
            logging.info(f"Model saved to {self.config.trained_model_path}")

            logging.info("Model training completed successfully")
            return metrics

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)


