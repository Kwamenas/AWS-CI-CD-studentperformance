from pathlib import Path
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer


@dataclass
#path definition
class DataTransformationConfig:
    #path to save my pickle file
    #we create artifacts
    #we create artifacts/pk
    artifact_dir : Path=Path("artifacts")
    preprocessor_object_file: Path=artifact_dir/"preprocessor.pkl" 


class DataTransformation:
    def __init__(self,filename: str ,data_path:str='artifacts'):

        self.data_path=Path(data_path)/filename
        self.data_transformation_config=DataTransformationConfig()

    def get_cols_preprocess (self):
        """
        This function will be responsible for getting the respective columns
        and preprocessing them
        """
        try:
            df=pd.read_csv(self.data_path)
            logging.info("Data loaded Sucessfully")

            cat_cols=df.select_dtypes(include='O').columns.to_list()
            numeric_cols=df.select_dtypes(exclude='O').drop(columns='math_score').columns.to_list()

            logging.info(f'Columns have been split into cat_cols{cat_cols} and numeric_cols{numeric_cols}')

            num_transformer=make_pipeline(SimpleImputer(strategy='mean'),StandardScaler())
            cat_transformer=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())

            feature_transformer=ColumnTransformer(
                transformers=[('num',num_transformer,numeric_cols),
                              ('cat',cat_transformer,cat_cols)
                              ]
            )

            return feature_transformer
        
        except Exception as e:
            raise CustomException(e)


        
    def initiate_data_transformation(self,train_path,valid_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_valid=pd.read_csv(valid_path)
            df_test=pd.read_csv(test_path)

            logging.info("Read and Loaded train data")
            logging.info("Read and Loaded valid data")
            logging.info("Read and Loaded test data")

            preprocessing_obj=self.get_cols_preprocess()

            X_train=df_train.drop(columns=['math_score'],axis=1)
            y_train=df_train["math_score"]

            X_valid=df_valid.drop(columns=['math_score'],axis=1)
            y_valid=df_valid["math_score"]

            X_test=df_test.drop(columns=['math_score'],axis=1)
            y_test=df_test["math_score"]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train=preprocessing_obj.fit_transform(X_train)
            X_valid=preprocessing_obj.transform(X_valid)
            X_test=preprocessing_obj.transform(X_test)

            train_trans=np.concatenate((X_train,np.array(y_train).reshape(-1,1)),axis=1)
            valid_trans=np.concatenate((X_valid,np.array(y_valid).reshape(-1,1)),axis=1)
            test_trans=np.concatenate((X_test,np.array(y_test).reshape(-1,1)),axis=1)

            logging.info(f"Saving perprocessing to a path")

            save_object(
                file_path=self.data_transformation_config.preprocessor_object_file,
                obj=preprocessing_obj
            )

            return(
                train_trans,
                valid_trans,
                test_trans,
                self.data_transformation_config.preprocessor_object_file
            )
        
        except Exception as e:
            raise CustomException(e)
        

if __name__=="__main__":
    transformer=DataTransformation(filename='train.csv')
    train_path="artifacts/train.csv"
    valid_path="artifacts/valid.csv"
    test_path="artifacts/test.csv"
    train_trans,valid_trans,test_trans,object_path =transformer.initiate_data_transformation(
        train_path,valid_path,test_path
    )