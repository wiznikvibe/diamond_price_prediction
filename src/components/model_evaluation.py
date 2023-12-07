import os, sys 
from typing import Optional
from src.logger import logging 
from src.exception import CustomException
from src.entity.config_entity import TRANSFORMER_OBJ_FILE_NAME, MODEL_FILE_NAME
from src.entity import config_entity, artifact_entity
from src.utils import load_object, load_numpy_arr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
import numpy as np


class ModelEvaluation:

    def __init__(self, model_trainer_artifact:artifact_entity.ModelTrainerArtifact,data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact

    def evaluate_metrics(self, actual, pred):
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)

        return r2, rmse, mae

    def initiate_model_evaluation(self,):

        try:
            test_array = load_numpy_arr(self.data_transformation_artifact.transform_test_dir)
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            print(f"Shape of data is : {test_array.shape}")
            model_path = self.model_trainer_artifact.model_obj_dir
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/wiznikvibe/diamond_price_prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)
            
            with mlflow.start_run():
                test_pred = model.predict(X_test)
                (r2, rmse, mae) = self.evaluate_metrics(y_test, test_pred)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            raise CustomException(e, sys)
