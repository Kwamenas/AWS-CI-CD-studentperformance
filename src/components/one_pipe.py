import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import src.logger
from src.exception import CustomException
from src.logger import logging

### now we need to create our artifacts directory
### this will house our data thus raw data, train, test,and valid



def data_ingestion(data_path: str, artifact_folder: str = "artifacts"):
    
    """Ingests data from CSV files in a given folder, concatenates them,
    and saves the raw dataset in an artifacts directory."""
    
    base_dir=Path(os.getcwd())
    ##NOW LET'S CREATE A DIRECTORY 
    logging.info('creating artifacts folder...')
    artifact_dir =base_dir/artifact_folder
    artifact_dir .mkdir(parents=True,exist_ok=True)

    # Build input data path
    data_input=base_dir/data_path/"data"

    files=list(data_input.glob("*.csv"))

    if len(files)==0:
        logging.info("There seems to be an error")
        raise CustomException(f"There is no CSV file in the path {str(data_input)}")
    
    logging.info(f"Found {len(files)} CSV files. Reading...")
    raw_data=[pd.read_csv(file)for file in files]
    df=pd.concat(raw_data,ignore_index=True)

    ##save the data to artefacts file
    raw_data_path=artifact_dir/"data.csv"
    df.to_csv(raw_data_path,index=False,header=True)
    logging.info(f"Raw data saved to {raw_data_path}")


    return df,artifact_dir, raw_data_path

def data_split(df: pd.DataFrame,artifact_dir:Path):

    logging.info("Starting train/valid/test split...")

    df_train,df_temp=train_test_split(df,test_size=0.2,random_state=42)
    df_valid,df_test=train_test_split(df_temp,test_size=0.5,random_state=42)
    
    train_data_path=artifact_dir/"train.csv"
    valid_data_path=artifact_dir/"valid.csv"
    test_data_path=artifact_dir/"test.csv"

    df_train.to_csv(train_data_path,index=False,header=True)
    df_valid.to_csv(valid_data_path,index=False,header=True)
    df_test.to_csv(test_data_path,index=False,header=True)

    logging.info(f"Train data saved to {train_data_path}")
    logging.info(f"Validation data saved to {valid_data_path}")
    logging.info(f"Test data saved to {test_data_path}")
    logging.info("Data split complete.")


    return df_train,df_valid,df_test

if __name__=="__main__":
    try:
        df,artifact_dir,raw_path=data_ingestion("notebook")
        print ("Sample top 5")
        print(f"Raw data saved into:{raw_path}")

        df_train,df_valid,df_test=data_split(df,artifact_dir)
        print("Train/Valid/Test split completed. Check artifacts folder.")
    except Exception as e:
        print("Error during ingestion:",e)




