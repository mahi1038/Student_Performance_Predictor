import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


class DataIngestionConfig:
    def __init__(self):
        self.train_data_path: str = os.path.join("dataset", "train_data.csv")
        self.test_data_path: str = os.path.join("dataset", "test_data.csv")
        self.raw_data_path: str = os.path.join("dataset", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        # read the raw data from the source
        raw_data = pd.read_csv(r'E:\environments\Student_Performance_Project\notebook\Student_data.csv')

        # creating dataset directory
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

        # saving the raw data 
        raw_data.to_csv(self.ingestion_config.raw_data_path)

        # train test split
        train_data, test_data = train_test_split(raw_data, train_size = 0.8, random_state = 42)

        train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
        test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path

        )
    
if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_ingestion()

    transformation = DataTransformation()
    train_array, test_array, _= transformation.initiate_transformation(train_path, test_path)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))

