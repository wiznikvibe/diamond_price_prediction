from dataclasses import dataclass 

@dataclass 
class DataIngestionArtifact:
    feature_store_dir: str 
    train_data_dir: str
    test_data_dir: str 

@dataclass 
class DataValidationArtifact:
    report_file_dir: str

@dataclass 
class DataTransformationArtifact:
    transformer_obj_dir: str
    transform_train_dir: str
    transform_test_dir: str

@dataclass 
class ModelTrainerArtifact:
    model_obj_dir: str