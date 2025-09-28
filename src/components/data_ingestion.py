import os
import sys
import src.logger
from src.exception import CustomException ##this gives us the custome exception
from src.logger import logging ###here we bring in the logger 
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#defining the class variables
#path for data storage
### actual class ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    valid_data_path: str=os.path.join('artifact',"valid.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(" Data Ingestion process")
        try:
            logging.info("Trying to read input dataset...")
            df=pd.read_csv('notebook//data//stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiation')
            train_set,temp_set=train_test_split(df,test_size=0.2,random_state=42)
            valid_set,test_set=train_test_split(temp_set,test_size=0.5,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            valid_set.to_csv(self.ingestion_config.valid_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.valid_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
           raise CustomException(e)


if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()