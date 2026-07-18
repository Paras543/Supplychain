import os
import sys

import pandas as pd
import joblib

from src.exception import CustomException
from src.logger import logging


class PredictionPipeline:
    def __init__(self):
        try:
            self.model = joblib.load(os.path.join('artifacts', 'xgb_model.pkl'))
            self.scaler = joblib.load(os.path.join('artifacts', 'scaler.pkl'))
            self.ohe = joblib.load(os.path.join('artifacts', 'onehot_encoder.pkl'))
            self.freq_maps = joblib.load(os.path.join('artifacts', 'freq_maps.pkl'))
            logging.info("Prediction pipeline artifacts loaded")
        except Exception as e:
            raise CustomException(e, sys)

        self.low_card_cols = [
            'Type', 'Customer Segment', 'Customer Country', 'Market',
            'Shipping Mode', 'Order Region', 'Department Name', 'Order Status'
        ]
        self.high_card_cols = [
            'Order City', 'Order State', 'Customer City',
            'Order Country', 'Product Name', 'Category Name', 'Customer State'
        ]

    def _preprocess(self, raw_row: dict) -> pd.DataFrame:
        df_row = pd.DataFrame([raw_row])

        
        df_row['order date (DateOrders)'] = pd.to_datetime(df_row['order date (DateOrders)'], format='mixed')
        df_row['order_month'] = df_row['order date (DateOrders)'].dt.month
        df_row['order_dow'] = df_row['order date (DateOrders)'].dt.dayofweek
        df_row['is_weekend'] = df_row['order_dow'].isin([5, 6]).astype(int)
        df_row = df_row.drop(columns=['order date (DateOrders)'])

        
        ohe_arr = self.ohe.transform(df_row[self.low_card_cols])
        ohe_df = pd.DataFrame(
            ohe_arr,
            columns=self.ohe.get_feature_names_out(self.low_card_cols),
            index=df_row.index
        )

        
        freq_df = pd.DataFrame(index=df_row.index)
        for col in self.high_card_cols:
            freq_df[col + '_freq'] = df_row[col].map(self.freq_maps[col]).fillna(0)

        
        remaining = df_row.drop(columns=self.low_card_cols + self.high_card_cols)
        full_row = pd.concat([remaining, ohe_df, freq_df], axis=1)

        
        full_row = full_row[self.model.get_booster().feature_names]

        
        scaled_row = pd.DataFrame(
            self.scaler.transform(full_row), columns=full_row.columns
        )
        return scaled_row

    def predict(self, raw_row: dict) -> dict:
        try:
            scaled_row = self._preprocess(raw_row)
            proba = self.model.predict_proba(scaled_row)[:, 1][0]
            pred = self.model.predict(scaled_row)[0]

            return {
                'risk_score': round(proba * 100, 1),
                'delay_probability': round(float(proba), 3),
                'prediction': 'Late' if pred == 1 else 'On-time'
            }
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)



