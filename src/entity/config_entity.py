import os, sys 
from datetime import datetime
from src.logger import logging 
from src.exception import CustomException

FILE_NAME = "gemstone.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJ_FILE_NAME = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"



class TrainingPipelineConfig: 

    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(),'artifact',f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

class DataIngestionConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.dataset_dir = os.path.join(os.getcwd(),"dataset", FILE_NAME)
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, "feature_store", FILE_NAME)
        self.train_data_dir = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
        self.test_data_dir = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
        self.test_size = 0.2

class DataValidationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_validation')
        self.report_file_dir = os.path.join(self.data_validation_dir, 'report.yaml')
        self.missing_value_threshold: float = 0.2
        self.base_file_dir = os.path.join(os.getcwd(),"dataset", FILE_NAME)

class DataTransformationConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.target_column = 'price'
        self.data_transformation_artifact = os.path.join(training_pipeline_config.artifact_dir,'data_transformation')
        self.transformer_obj_dir = os.path.join(self.data_transformation_artifact, 'transformer', TRANSFORMER_OBJ_FILE_NAME)
        self.transform_train_dir = os.path.join(self.data_transformation_artifact, 'transformer', TRAIN_FILE_NAME.replace('csv', 'npz'))
        self.transform_test_dir = os.path.join(self.data_transformation_artifact, 'transformer', TEST_FILE_NAME.replace('csv', 'npz'))
        
class ModelTrainerConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_artifact = os.path.join(training_pipeline_config.artifact_dir,'model_trainer')
        self.model_obj_dir = os.path.join(self.model_trainer_artifact, MODEL_FILE_NAME)