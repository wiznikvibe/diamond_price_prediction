import os, sys 
from typing import Optional
from src.logger import logging 
from src.exception import CustomException
from src.entity.config_entity import TRANSFORMER_OBJ_FILE_NAME, MODEL_FILE_NAME
from src.entity import config_entity, artifact_entity
from src.utils import load_object
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
import numpy as np


class ModelEvaluation:

    def __init__(self, model_eval_config: config_entity.ModelEvaluationConfig):
        self.model_eval_config = model_eval_config
        

    def initiate_model_evaluation(self):
        pass 
