from dataclasses import dataclass 

@dataclass 
class DataIngestionArtifact:
    feature_store_dir: str 
    train_data_dir: str
    test_data_dir: str 

@dataclass 
class DataValidationArtifact:
    report_file_dir