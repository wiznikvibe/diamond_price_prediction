from src.entity import config_entity
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import os, sys




if __name__ == "__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        config = config_entity.DataIngestionConfig(training_pipeline_config)
        print(config.dataset_path)
        df = pd.read_csv(config.dataset_path)
        print(df.shape)
        logging.info("First Basic Setup")

    except Exception as e:
        raise CustomException(e, sys)