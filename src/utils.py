import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle as pk
import dill
from sklearn.metrics import r2_score
import pandas as pd 
from pandas import DataFrame

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=Path(file_path).parent
        dir_path.mkdir(parents=True,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pk.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e)

def train_evaluate_model(X_train,y_train,X_valid,y_valid,X_test,y_test,models):
    try:
        model_dict={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model_name=list(models.keys())[i]
            model.fit(X_train,y_train)

    #make predictions
            y_train_pred=model.predict(X_train)
            y_valid_pred=model.predict(X_valid)
            y_test_pred=model.predict(X_test)

            trained_r2_score=r2_score(y_train,y_train_pred)
            valid_r2_score=r2_score(y_valid,y_valid_pred)
            test_r2_score=r2_score(y_test,y_test_pred)

            model_dict[model_name] = {
                'Trained_R2_Score': round(trained_r2_score, 4),
                'Valid_R2_Score': round(valid_r2_score, 4),
                'Test_R2_Score': round(test_r2_score, 4)
                }
        return DataFrame(model_dict).T.sort_values(by='Test_R2_Score',ascending=False)

    except Exception as e:
        raise CustomException(e)

