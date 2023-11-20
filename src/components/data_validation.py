import os, sys
import pandas as pd 
import numpy as np
from typing import Optional
from src.logger import logging
from src.exception import CustomException
from src.utils import write_yaml_file
from scipy.stats import ks_2samp
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

class DataValidation:

    def __init__(self, data_validation_config:DataValidationConfig, data_ingestion_artifact:DataIngestionArtifact):
        logging.info(f"{'='*20} Data Validation  {'='*20}")
        print(f"{'='*20} Data Validation  {'='*20}")
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.validation_info = dict()

    def drop_missing_values_columns(self, df:pd.DataFrame, report_key:str)->Optional[pd.DataFrame]:
        try:
            thresh = self.data_validation_config.missing_value_threshold
            logging.info(f"Dropping Columns with missing values over the threshold:{thresh}")
            null_report = df.isnull().sum()
            drop_column_list = null_report[null_report>thresh].index
            logging.info(f"Columns with Missing Values: {drop_column_list}")
            df.drop(drop_column_list, axis=1, inplace=True)
            self.validation_info[report_key] = drop_column_list
            if len(df.columns) == 0:
                logging.info("Invalid Dataset")
                return None
            return df 
            
        except Exception as e:
            raise CustomException(e, sys)

    
    def is_required_column_exists(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key:str)->bool:
        try: 
            logging.info("Checking for Required Columns")
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for column in base_columns:
                if column not in current_columns:
                    missing_columns.append(column)
            if len(missing_columns) > 0:
                self.validation_info[report_key] = missing_columns
                logging.info(f"Missing Columns: {missing_columns}")
                return False
            
            return True 
        except Exception as e:
            raise CustomException(e, sys)


    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key: str):
        try:
            drift_record = dict()
            base_columns = base_df.columns
            current_columns = current_df.columns
            logging.info("==================Checking for Data Drift===============")
            for column in base_columns:
                base_data, current_data = base_df[column], current_df[column]
                column_distribution = ks_2samp(base_data, current_data)

                if column_distribution.pvalue > 0.05:
                    drift_record[column] = {
                        'p_value': column_distribution.pvalue,
                        'same_distribution': True
                    }
                else:
                    drift_record[column] = {
                        'p_value': column_distribution.pvalue,
                        'same_distribution': False
                    }
            self.validation_info[report_key] = drift_record
            logging.info(f"Drift Report: {drift_record}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self)-> DataIngestionArtifact:
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_dir)
            base_df.replace({'nan':np.NAN}, inplace=True)
            base_df = self.drop_missing_values_columns(df=base_df, report_key="Base_Data_Missing")

            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_dir)
            train_df.replace({'nan':np.NAN}, inplace=True)
            train_df = self.drop_missing_values_columns(df=train_df, report_key="Train_Data_Missing")

            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_dir)
            test_df.replace({'nan':np.NAN}, inplace=True)
            test_df = self.drop_missing_values_columns(df=test_df, report_key="Test_Data_Missing")

            train_status = self.is_required_column_exists(base_df=base_df, current_df=train_df, report_key='missing_columns_training_data')
            test_status = self.is_required_column_exists(base_df=base_df, current_df=test_df, report_key='missing_columns_testing_data')

            if train_status:
                self.data_drift(base_df=base_df, current_df=train_df, report_key='train_data_drift')

            if test_status:
                self.data_drift(base_df=base_df, current_df=test_df, report_key='test_data_drift')

            logging.info("Generating Validation Report.")
            print("Generating Validation Report.")

            write_yaml_file(file_path=self.data_validation_config.report_file_dir, data=self.validation_info)
            data_validation_artifact = DataValidationArtifact(report_file_dir=self.data_validation_config.report_file_dir)
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)


