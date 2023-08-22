import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def convert_milage_to_int(self, milage):
        new_milage = float(milage.split(' ')[0])
        if milage.split(' ')[1] == 'km/kg':
            new_milage = new_milage * 0.74

        return new_milage

    def data_preprocessing(self, data):

        data.drop('Unnamed: 0', axis=1, inplace=True)
        data.drop('New_Price', axis=1, inplace=True)
        data.dropna(subset='Mileage', inplace=True)

        # Convert Milage to int
        data['Mileage'] = data['Mileage'].apply(self.convert_milage_to_int)

        data.Seats = data.Seats.replace(0, np.nan)
        data['Engine'] = data['Engine'].str.rstrip(' CC')
        data['Power'] = data['Power'].str.rstrip(' bhp')
        data['Engine'] = data['Engine'].astype('float')
        data['Power'] = pd.to_numeric(data['Power'], errors='coerce')
        data['brand'] = data['Name'].str.upper().str.split(' ').str[0]
        data.drop('Name', axis=1, inplace=True)

        return data

    def get_data_tranformer_object(self):
        '''
        This function is responsible for the data transformation

        '''
        try:
            numerical_columns = [
                'Year',
                'Kilometers_Driven',
                'Mileage',
                'Engine',
                'Power',
                'Seats'
            ]
            categorical_columns = [
                'Location',
                'Fuel_Type',
                'Transmission',
                'Owner_Type',
                'brand'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scalar", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns   : {numerical_columns}")
            logging.info(f"Categorical columns : {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = self.data_preprocessing(pd.read_csv(train_path))
            test_df = self.data_preprocessing(pd.read_csv(test_path))

            logging.info("Read Train and Test data completed")

            logging.info("Obtaining preprocessing objects")

            prepocessing_obj = self.get_data_tranformer_object()

            target_column_name = "Price"

            input_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = prepocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = prepocessing_obj.transform(
                input_feature_test_df)

            # train_arr = np.c_[input_feature_train_arr.toarray(),
            #                   np.array(target_feature_train_df).reshape(1, -1)]
            # test_arr = np.c_[input_feature_test_arr.toarray(),
            #                  np.array(target_feature_test_df).reshape(1, -1)]

            train_arr = np.hstack((input_feature_train_arr.toarray(
            ), np.array(target_feature_train_df).reshape(-1, 1)))
            test_arr = np.hstack((input_feature_test_arr.toarray(
            ), np.array(target_feature_test_df).reshape(-1, 1)))

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=prepocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    path1 = os.path.join('artifacts', 'train.csv')
    path2 = os.path.join('artifacts', 'test.csv')
    data = DataTransformation()
    data.initiate_data_transformation(path1, path2)
