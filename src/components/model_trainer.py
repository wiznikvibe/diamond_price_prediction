import os, sys 
import numpy as np
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig, data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def evaluate_regression(self,predicted, true):
        mse = mean_squared_error(predicted, true)
        mae = mean_absolute_error(predicted, true)
        r2 = r2_score(predicted, true)

        return mse, mae, r2 

    def evaluate_models(self, X, y, models): 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=30)
        models_list = []
        score_list = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            
            # model_train_mse, model_train_mae, model_train_r2 = evaluate_regression(y_train_pred, y_train)
            model_test_mse, model_test_mae, model_test_r2 = evaluate_regression(y_test_pred, y_test)

            
            
            
            print(model_name)
            models_list.append(model_name)
            # print('Model performance for Training set')
            # print('- Mean Squared Error (MSE): {:.4f}'.format(model_train_mse))
            # print('- Mean Absolute Error (MAE): {:.4f}'.format(model_train_mae))
            # print('- R-squared: {:.4f}'.format(model_train_r2))
            

            print('----------------------------------')

            print('Model performance for Test set')
            print('- Mean Squared Error (MSE): {:.4f}'.format(model_test_mse))
            print('- Mean Absolute Error (MAE): {:.4f}'.format(model_test_mae))
            print('- R-squared: {:.4f}'.format(model_test_r2 * 100))

            score_list.append(model_test_r2)
            print('=='*20)
        
        report = pd.DataFrame(list(zip(models_list, score_list)), columns=['Model Name', 'R2_Score']).sort_values(by=['R2_Score'], ascending=False)
        return report


