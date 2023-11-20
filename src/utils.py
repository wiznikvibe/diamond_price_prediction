import os, sys 
import yaml, dill 
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
