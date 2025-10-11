from src.exception import CustomException ##this gives us the custome exception
from src.logger import logging ###here we bring in the logger 
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

#defining the class variables
#path for data storage
### actual class ingestion
@dataclass
class DataIngestionConfig:

    artifact_dir: Path = Path("artifacts")
    train_data_path: Path=artifact_dir/"train.csv"
    valid_data_path: Path=artifact_dir/"valid.csv"
    test_data_path: Path=artifact_dir/"test.csv"
    raw_data_path: Path=artifact_dir/"data.csv"


class DataIngestion:
    def __init__(self,data_path:str):

        self.data_path=Path(data_path)/"data" ### telling us where the data is 
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self) -> pd.DataFrame:
        logging.info(" Data Ingestion process begin....")
        try:
            logging.info("Creating artifact directory")
            self.ingestion_config.artifact_dir.mkdir(parents=True,exist_ok=True)

            files=list(self.data_path.glob("*.csv"))
            if not files:
                raise CustomException(f"No CSV file in {self.data_path}")
            
            logging.info(f"Found {len(files)} CSV file(s). Reading........")
            dfs=[pd.read_csv(file) for file  in files ]
            df=pd.concat(dfs,ignore_index=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f'Raw data saved to {self.ingestion_config.raw_data_path} ')

            return df


        except Exception as e:
            raise CustomException(e)
        

    def data_split(self,df: pd.DataFrame):
            
            try:
                
                logging.info('Train test split initiation')
                train_set,temp_set=train_test_split(df,test_size=0.2,random_state=42)
                valid_set,test_set=train_test_split(temp_set,test_size=0.5,random_state=42)

                train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
                valid_set.to_csv(self.ingestion_config.valid_data_path,index=False,header=True)
                test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

                logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
                logging.info(f"Validation data saved to {self.ingestion_config.valid_data_path}")
                logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
                logging.info("Data split complete.")

                return train_set,valid_set,test_set
            
            except Exception as e:
                raise CustomException(e)


##if __name__=="__main__":
##    try:
 ##       ingestion=DataIngestion("notebook")
 ##       df=ingestion.initiate_data_ingestion()

   ##     print("Raw data preview:")
     ##   print(df.head())

       ## train_set,valid_set,test_set=ingestion.data_split(df)
       ## print("Data sucessfully split and saved")

        ##transformer=DataTransformation(filename="train.csv")
        ##train_path="artifacts/train.csv"
        ##valid_path="artifacts/valid.csv"
        ##test_path="artifacts/test.csv"

        ##train_trans,valid_trans,test_trans,object_path=transformer.initiate_data_transformation(
          ##  train_path,valid_path,test_path
        ##)

        ##model_trainer=ModelTrainer()
        ##print(model_trainer.train_model(train_trans,valid_trans,test_trans))


    ##except Exception as e:
      ##  print("Error",e)
            