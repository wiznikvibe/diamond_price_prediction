import os, sys 
from src.logger import logging
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
# from src.components.model_pusher import FinalModelConfig

def start_training_pipeline():
    try:
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
        # print(data_transformation_artifact)

        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()  
        # print(model_trainer_artifact)

        model_evaluation = ModelEvaluation(model_trainer_artifact=model_trainer_artifact,data_transformation_artifact=data_transformation_artifact)
        model_evaluation.initiate_model_evaluation()
    except Exception as e:
        raise CustomException(e, sys)
