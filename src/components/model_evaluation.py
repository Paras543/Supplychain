import os
import sys
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelEvaluationConfig:
    report_path: str = os.path.join('artifacts', 'classification_report.txt')
    confusion_matrix_path: str = os.path.join('artifacts', 'confusion_matrix.csv')


class ModelEvaluation:
    def __init__(self):
        self.config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, model, X_test, y_test):
        logging.info("Model evaluation started")
        try:
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, target_names=['On-time', 'Late'])
            logging.info(f"Classification report:\n{report}")

            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=['Actual: On-time', 'Actual: Late'],
                columns=['Pred: On-time', 'Pred: Late']
            )
            logging.info(f"Confusion matrix:\n{cm_df}")

            os.makedirs('artifacts', exist_ok=True)
            with open(self.config.report_path, 'w') as f:
                f.write(report)
            cm_df.to_csv(self.config.confusion_matrix_path)

            logging.info("Model evaluation completed successfully")
            return report, cm_df

        except Exception as e:
            logging.error("Error occurred during model evaluation")
            raise CustomException(e, sys)


