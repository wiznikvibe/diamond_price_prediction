from src.entity import config_entity
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import os, sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from src.entity import artifact_entity
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

training_pipeline_config = config_entity.TrainingPipelineConfig()

data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
data_validation_artifact = data_validation.initiate_data_validation()
# print(data_validation_artifact)

data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
data_transformation_artifact = data_transformation.initiate_data_transformation()
print(data_transformation_artifact)




