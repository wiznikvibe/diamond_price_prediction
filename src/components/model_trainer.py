import os, sys 
import numpy as np
import pandas as pd 
from src.logger import logging
from src.utils import save_object, load_object, load_numpy_arr
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig, data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def evaluate_regression(self,predicted, true):
        mse = mean_squared_error(predicted, true)
        mae = mean_absolute_error(predicted, true)
        r2 = r2_score(predicted, true)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        return  r2 

    def train_model(self, X, y):
        try:
            model = GradientBoostingRegressor()
            model.fit(X, y)
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            train_array = load_numpy_arr(self.data_transformation_artifact.transform_train_dir)
            test_array = load_numpy_arr(file_dir=self.data_transformation_artifact.transform_test_dir)
            
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            model = self.train_model(X=X_train,y=y_train)
            logging.info(f"Model is trained on these features: {model.feature_importances_}")

            logging.info("Computing R2 Score for Train Data")
            y_train_pred = model.predict(X_train)
            r2_train_score = self.evaluate_regression(predicted=y_train_pred, true=y_train)

            logging.info("Computing R2 Score for Test Data")
            y_test_pred = model.predict(X_test)
            r2_test_score = self.evaluate_regression(predicted=y_test_pred, true=y_test)

            logging.info(f"R2 Score for Train Data: {r2_train_score} || R2 Score for Test Data: {r2_test_score}")
            print(f"R2 Score for Train Data: {r2_train_score} || R2 Score for Test Data: {r2_test_score}")

            save_object(file_dir=self.model_trainer_config.model_obj_dir, obj=model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_obj_dir=self.model_trainer_config.model_obj_dir
            )

            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
            



            
            
            
            
            


    


