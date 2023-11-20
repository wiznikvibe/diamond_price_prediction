import os, sys 
from src.logger import logging
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from src.exception import CustomException
from src.entity import artifact_entity, config_entity

class DataIngestion:
    
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        print(f"{'='*40} Loading Data Ingestion... {'='*40}")
        logging.info(f"{'='*40} Loading Data Ingestion... {'='*40}")
        self.data_ingestion_config = data_ingestion_config 

    def initiate_data_ingestion(self, )-> artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Extracting dataset from the database.")
            data = pd.read_csv(self.data_ingestion_config.dataset_dir)
            logging.info(f"Dataset extraction complete, Shape of the data: {data.shape}")
            
            data.replace(to_replace='na', value=np.NAN, inplace=True)
            logging.info("Replacing Null values with Numpy obj")

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_dir)
            os.makedirs(feature_store_dir, exist_ok=True)

            data.to_csv(path_or_buf=self.data_ingestion_config.feature_store_dir, index=False, header=True)
            logging.info("Storing raw files into the Feature Directory..")

            logging.info("Spliting the Training and Testing datasets.")
            train_df, test_df = train_test_split(data, test_size=self.data_ingestion_config.test_size, random_state=42)

            dataset_dir = os.path.dirname(self.data_ingestion_config.train_data_dir)
            os.makedirs(dataset_dir, exist_ok=True)

            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_data_dir, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_data_dir, index=False, header=True)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_dir=self.data_ingestion_config.feature_store_dir,
                train_data_dir=self.data_ingestion_config.train_data_dir,
                test_data_dir=self.data_ingestion_config.test_data_dir
            ) 

            logging.info(f"TrainDataSize:{train_df.shape} || TestDataSize: {test_df.shape}")
            logging.info(f"{'='*20} Exiting Data Ingestion.. {'='*20}")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)