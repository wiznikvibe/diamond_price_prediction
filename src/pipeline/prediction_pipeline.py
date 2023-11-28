import os, sys 
from datetime import datetime
import pandas as pd 
from src.entity.config_entity import MODEL_FILE_NAME, TRANSFORMER_OBJ_FILE_NAME
from src.logger import logging 
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    
    def __init__(self,):
        self.latest_dir = self.get_latest_directory(r"C:/Users/nikhi/diamond_price_prediction/artifact")

    
    
    def get_latest_directory(self, file_dir:str):
        try:
            all_dirs = [d for d in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, d))]
            date_objs = [datetime.strptime(d,'%m%d%Y__%H%M%S') for d in all_dirs]

            latest_date = max(date_objs)
            latest_directory = latest_date.strftime('%m%d%Y__%H%M%S')
            return os.path.join(file_dir, latest_directory)
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys) 
        
    def predict(self, features):
        try:
            transformer_obj_path = os.path.join(self.latest_dir, "data_transformation", "transformer", TRANSFORMER_OBJ_FILE_NAME)
            model_obj_path = os.path.join(self.latest_dir, "model_trainer", MODEL_FILE_NAME)
            preprocessor = load_object(transformer_obj_path)
            model = load_object(model_obj_path)
            

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

