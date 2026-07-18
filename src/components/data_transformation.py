import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_scaler_path: str = os.path.join('artifacts', 'scaler.pkl')
    preprocessor_ohe_path: str = os.path.join('artifacts', 'onehot_encoder.pkl')  # fixed: was 'ohe.pkl'
    preprocessor_freq_path: str = os.path.join('artifacts', 'freq_maps.pkl')      # fixed: was 'freq_map.pkl'

    low_card_cols: tuple = (
        'Type', 'Customer Segment', 'Customer Country', 'Market',
        'Shipping Mode', 'Order Region', 'Department Name', 'Order Status'
    )
    high_card_cols: tuple = (
        'Order City', 'Order State', 'Customer City',
        'Order Country', 'Product Name', 'Category Name', 'Customer State'
    )
    id_cols: tuple = (
        'Order Id', 'Order Item Id', 'Order Customer Id', 'Order Item Cardprod Id'
    )
    target_col: str = 'Late_delivery_risk'


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()  # fixed: was DataTransformation() — infinite recursion

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract month, day-of-week, and weekend flag from the order date column."""
        df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], format='mixed')
        df['order_month'] = df['order date (DateOrders)'].dt.month
        df['order_dow'] = df['order date (DateOrders)'].dt.dayofweek
        df['is_weekend'] = df['order_dow'].isin([5, 6]).astype(int)
        df = df.drop(columns=['order date (DateOrders)'])
        return df

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # fixed: was train_path = pd.read_csv(train_path) — wrong variable name
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Drop ID columns that are present in the dataframe
            id_cols_present = [c for c in self.config.id_cols if c in train_df.columns]
            train_df = train_df.drop(columns=id_cols_present)
            test_df = test_df.drop(columns=id_cols_present)

            train_df = self._add_calendar_features(train_df)
            test_df = self._add_calendar_features(test_df)
            logging.info("Calendar features added")

            target = self.config.target_col
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            low_card = list(self.config.low_card_cols)
            high_card = list(self.config.high_card_cols)

            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe.fit(X_train[low_card])  # fixed: was ohe.fit[...] — square bracket typo

            train_ohe = pd.DataFrame(
                ohe.transform(X_train[low_card]),
                columns=ohe.get_feature_names_out(low_card),
                index=X_train.index
            )
            test_ohe = pd.DataFrame(
                ohe.transform(X_test[low_card]),
                columns=ohe.get_feature_names_out(low_card),
                index=X_test.index
            )
            logging.info('One hot encoding done')

            freq_maps = {}
            train_freq = pd.DataFrame(index=X_train.index)
            test_freq = pd.DataFrame(index=X_test.index)

            for col in high_card:
                freq_maps[col] = X_train[col].value_counts(normalize=True)
                train_freq[col + '_freq'] = X_train[col].map(freq_maps[col])
                test_freq[col + '_freq'] = X_test[col].map(freq_maps[col]).fillna(0)
            logging.info("Frequency encoding complete")

            all_cat_cols = low_card + high_card
            X_train_final = pd.concat(
                [X_train.drop(columns=all_cat_cols), train_ohe, train_freq], axis=1
            )
            X_test_final = pd.concat(
                [X_test.drop(columns=all_cat_cols), test_ohe, test_freq], axis=1
            )
            logging.info(f"Combined feature shape: train {X_train_final.shape}, test {X_test_final.shape}")

            scaler = StandardScaler()
            # fixed: was scaler.fit_transform(X_train, y_train) — wrong variable + StandardScaler ignores y
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_final),
                columns=X_train_final.columns,
                index=X_train_final.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_final),
                columns=X_test_final.columns,
                index=X_test_final.index
            )
            logging.info("Scaling complete")

            os.makedirs('artifacts', exist_ok=True)
            joblib.dump(scaler, self.config.preprocessor_scaler_path)
            joblib.dump(ohe, self.config.preprocessor_ohe_path)
            joblib.dump(freq_maps, self.config.preprocessor_freq_path)
            logging.info("Preprocessor artifacts saved")

            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            logging.error("Exception occurred in data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(
        source_path="data/raw/DataCoSupplyChainDataset.csv"
    )

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
        train_path, test_path
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            






            
