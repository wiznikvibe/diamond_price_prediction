import os, sys 
import yaml, dill 
import numpy as np
import pandas as pd
from src.logger import logging 
from src.exception import CustomException

def write_yaml_file(file_path, data:dict):
    try:
        logging.info("Creating Validation Report")
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
            file.close() 
            
    except Exception as e:
        raise CustomException(e, sys)

def save_numpy_arr(file_dir: str, array:np.array)->None:
    try: 
        logging.info("Saving the Transformered data inside a Numpy Object")
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        with open(file_dir, 'wb') as file_obj:
            np.save(file_obj, array)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_numpy_arr(file_dir: str):
    try:
        if not os.path.exists(file_dir):
            raise Exception("Numpy File does not exists")
        with open(file_dir, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
            
def save_object(file_dir: str, obj:object)->None:
    try:
        logging.info("Saving the Transformer Object for Future Reference")
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        with open(file_dir, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_dir: str):
    try:
        if not os.path.exists(file_dir):
            raise Exception(f'File Directory:{file_dir} Does not exist')
        with open(file_dir, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# def get_latest_dir(file_dir: str):
#     try:
#         if not os.path.exists(file_dir):
#             raise Exception("No Such File Directory")
#         latest_folder = max(os.listdir(file_dir))
