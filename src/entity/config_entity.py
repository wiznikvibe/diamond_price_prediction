import os, sys 
from datetime import datetime
from src.logger import logging 
from src.exception import CustomException

FILE_NAME = "gemstone.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
MODEL_FILE_NAME = "model.pkl"
TRANSFORMER_OBJ_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJ_FILE_NAME = "target_encoder.pkl"


class TrainingPipelineConfig: 

    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(),'artifact',f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

class DataIngestionConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_artifact = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
        self.dataset_path = os.path.join(os.getcwd(),"dataset", FILE_NAME)
