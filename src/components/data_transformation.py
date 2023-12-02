import os, sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.utils import save_object, save_numpy_arr
from src.exception import CustomException
from src.entity import config_entity, artifact_entity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler




class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        print(f"{'='*20} Data Transformation {'='*20}")
        logging.info(f"{'='*20} Data Transformation {'='*20}")
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact


    @classmethod
    def get_data_transformer_object(cls)-> Pipeline:
        try:
            logging.info("Data Transformation Initiated")

            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'x', 'y', 'z']

            # Ordinal Ranking Per Category
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) 


    def outlier_treatment(self, df:pd.DataFrame, columns:list, multiplier:float=1.5)-> pd.DataFrame:
        try:
            treated_df = df.copy()
            for col in columns:
                q1 = treated_df[col].quantile(0.25)
                q3 = treated_df[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - multiplier*iqr
                upper_bound = q3 + multiplier*iqr

                treated_df[col] = treated_df[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
            return treated_df
        except Exception as e:
            raise CustomException(e, sys)
            
            
    def initiate_data_transformation(self)-> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_dir)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_dir)
            numerical_columns = train_df.columns[train_df.dtypes != 'object']
            outlier_treatment_train_df = self.outlier_treatment(df=train_df, columns=numerical_columns)
            outlier_treatment_test_df = self.outlier_treatment(df=test_df, columns=numerical_columns)


            logging.info(f"Reading the Train and Test Data, Train: {train_df.shape}|| Test: {test_df.shape}")
            logging.info(f"Reading the Train and Test Data, Train: {train_df.columns}|| Test: {test_df.columns}")

            columns_to_drop = ['id' ,'depth', 'table', 'price']
            
            input_features_train_df = outlier_treatment_train_df.drop(columns_to_drop, axis=1, inplace=False)
            logging.info(f"Input Features: {input_features_train_df.columns}")
            target_feature_train_df = outlier_treatment_train_df[self.data_transformation_config.target_column]

            input_features_test_df = outlier_treatment_test_df.drop(columns_to_drop, axis=1, inplace=False)

            target_feature_test_df = outlier_treatment_test_df[self.data_transformation_config.target_column]

            preprocessor = DataTransformation.get_data_transformer_object()

            

            input_feature_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor.transform(input_features_test_df)

            logging.info(f'Preprocessor Features: {preprocessor.get_feature_names_out()}')
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            save_object(file_dir=self.data_transformation_config.transformer_obj_dir, obj=preprocessor)
            save_numpy_arr(file_dir=self.data_transformation_config.transform_train_dir, array=train_arr)
            save_numpy_arr(file_dir=self.data_transformation_config.transform_test_dir, array=test_arr)

            logging.info("Preprocessor Pickle & Array Saved")

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformer_obj_dir=self.data_transformation_config.transformer_obj_dir,
                transform_train_dir=self.data_transformation_config.transform_train_dir,
                transform_test_dir=self.data_transformation_config.transform_test_dir
            )

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) 

