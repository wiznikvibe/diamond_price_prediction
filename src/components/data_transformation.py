import os, sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.utils import save_object, save_numpy_arr
from src.exception import CustomException
from src.entity import config_entity, artifact_entity


class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        print(f"{'='*20} Data Transformation {'='*20}")
        logging.info(f"{'='*20} Data Transformation {'='*20}")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact


    @classmethod
    def get_data_transformer_object(cls)-> Pipeline:
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) 

    def initiate_data_transformation(self)-> artifact_entity.DataTransformationArtifact:
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) 

