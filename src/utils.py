import os
import sys
import numpy as np
from pathlib import Path
import pickle as pk
from sklearn.metrics import r2_score
import pandas as pd 
from pandas import DataFrame
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=Path(file_path).parent
        dir_path.mkdir(parents=True,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pk.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e)

def train_evaluate_model(X_train,y_train,X_valid,y_valid,X_test,y_test,models,params,n_iter=2,cv=3):
    try:
        model_dict={}
        ##for i in range(len(list(models))):
          ##  model=list(models.values())[i]
            ##model_name=list(models.keys())[i]
        for model_name,model in models.items():
            rsc=RandomizedSearchCV(estimator=model,
                                   param_distributions=params[model_name],n_iter=n_iter
                                   ,cv=cv,verbose=2,n_jobs=-1)
            rsc.fit(X_train,y_train)
            model.set_params(**rsc.best_params_)
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

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pk.load(file_obj)

    except Exception as e:
        raise CustomException(e)
