import os, sys 
import numpy as np
import pandas as pd 
from src.logger import logging
from src.utils import save_object, load_object, load_numpy_arr
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

    def evaluate_models(self, X_train, y_train, X_test, y_test, models)-> pd.DataFrame: 

        models_list = []
        score_list = []

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            
            model_test_mse, model_test_mae, model_test_r2 = self.evaluate_regression(y_test_pred, y_test)

            print(model_name)
            models_list.append(model_name)
            
            print('----------------------------------')

            print('Model performance for Test set')
            print('- Mean Squared Error (MSE): {:.4f}'.format(model_test_mse))
            print('- Mean Absolute Error (MAE): {:.4f}'.format(model_test_mae))
            print('- R-squared: {:.4f}'.format(model_test_r2 * 100))
            
            logging.info('- Mean Squared Error (MSE): {:.4f}'.format(model_test_mse))
            logging.info('- Mean Absolute Error (MAE): {:.4f}'.format(model_test_mae))
            logging.info('- R-squared: {:.4f}'.format(model_test_r2 * 100))

            score_list.append(model_test_r2)
            print('=='*20)
        
        report = pd.DataFrame(list(zip(models_list, score_list)), columns=['Model_Name', 'R2_Score']).sort_values(by=['R2_Score'], ascending=False)
        return report

    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            train_array = load_numpy_arr(self.data_transformation_artifact.transform_train_dir)
            test_array = load_numpy_arr(file_dir=self.data_transformation_artifact.transform_test_dir)
            
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor()
                
            } 

            model_report = self.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models)
            logging.info(model_report)
            
            best_model = model_report[model_report.R2_Score == model_report.R2_Score.max()]
            print(f"Best Model:{best_model['Model_Name']} || Best Model Score: {best_model['R2_Score']}")
            logging.info(f"Best Model:{best_model['Model_Name']} || Best Model Score: {best_model['R2_Score']}")
            best_model_name = best_model['Model_Name'].values[0]
            save_model = models[best_model_name] 

            save_object(file_dir=self.model_trainer_config.model_obj_dir, obj=save_model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_obj_dir=self.model_trainer_config.model_obj_dir
            )

            return model_trainer_artifact

            
            
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
            


    


