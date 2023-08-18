import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Location: str,
                 Fuel_Type: str,
                 Transmission: str,
                 Owner_Type: str,
                 brand: str,
                 Year: int,
                 Kilometers_Driven: int,
                 Mileage: int,
                 Engine: int,
                 Power: int,
                 Seats: int):

        self.Location = Location
        self.Fuel_Type = Fuel_Type
        self.Transmission = Transmission
        self.Owner_Type = Owner_Type
        self.brand = brand
        self.Year = Year
        self.Kilometers_Driven = Kilometers_Driven
        self.Mileage = Mileage
        self.Engine = Engine
        self.Power = Power
        self.Seats = Seats

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Location": [self.Location],
                "Fuel_Type": [self.Fuel_Type],
                "Transmission": [self.Transmission],
                "Owner_Type": [self.Owner_Type],
                "brand": [self.brand],
                "Year": [self.Year],
                "Kilometers_Driven": [self.Kilometers_Driven],
                "Mileage": [self.Mileage],
                "Engine": [self.Engine],
                "Power": [self.Power],
                "Seats": [self.Seats]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
