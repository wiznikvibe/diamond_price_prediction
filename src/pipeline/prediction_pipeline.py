import os, sys 
import pandas as pd 
from src.logger import logging 
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    
    def __init__(self):
        self.base_directory = os.path.join('artifact', max(os.listdir('artifact')))
        self.transformer_obj_path = os.path.join(self.base_directory,'data_transformation','transformer','transformer.pkl')
        self.model_obj_path = os.path.join(self.base_directory,'model_trainer', 'model.pkl')
        
    def predict(self, features):
        try:
            preprocessor = load_object(self.transformer_obj_path)
            model = load_object(self.model_obj_path)
            

            scaled = preprocessor.transform(features)
            pred = model.predict(scaled)
            return pred 
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, carat: float, cut: str, color: str, clarity: str, x: float, y: float, z: float):
        self.carat = carat
        self.cut = cut 
        self.color = color
        self.clarity = clarity
        self.x = x 
        self.y = y 
        self.z = z

    def get_as_dataframe(self):
        try: 
            custom_data_input_dict = {
                'carat': [self.carat],
                'cut' : [self.cut],
                'color' : [self.color],
                'clarity' : [self.clarity],
                'x' : [self.x],
                'y' : [self.y],
                'z' : [self.z],
            } 

            df = pd.DataFrame(custom_data_input_dict)
            return df 
        except Exception as e:
            raise CustomException(e, sys)

