import pandas as pd
import numpy as np
import pickle as pk
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
from pathlib import Path


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            ##load the path thus for model and preprocessor
            ## then transform and predict
            model_path=Path('artifacts/model.pkl')
            preprocess_path=Path('artifacts/preprocessor.pkl')
            logging.info("Loading model and preprocessor...")
            print("Loading model and pre-processor")

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocess_path)
            logging.info("Loaded the model and preprocessor successfully")

            print("Sucessfully loaded model and preprocessor ")
            data_sc=preprocessor.transform(features)
            preds=model.predict(data_sc)

            return preds
            

        except CustomException as e:
            raise CustomException(e)
        
class CustomData:
    def __init__(self,
                 gender:str,race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
        
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score


    
    def trans_dataframe(self):
        try:
            custom_data_dict={
                'gender':self.gender,
                'race_ethnicity':self.race_ethnicity,
                'parental_level_of_education':self.parental_level_of_education,
                'lunch':self.lunch,
                'test_preparation_course':self.test_preparation_course,
                'reading_score':self.reading_score,
                'writing_score':self.writing_score

            }

            return pd.DataFrame([custom_data_dict])
        except CustomException as e:
            raise CustomException(e)

        

