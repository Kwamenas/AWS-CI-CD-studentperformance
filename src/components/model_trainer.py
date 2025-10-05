from pathlib import Path
from src.utils import save_object,train_evaluate_model
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

##create the model trainer config
@dataclass
class ModelTrainerConfig:
    artifact_dir : Path=Path("artifacts")
    model_object_file: Path=artifact_dir/"model.pkl" 

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def train_model(self,train_trans,valid_trans,test_trans):
        try:
            logging.info("Re-spliting into features and target")
            X_train,y_train,X_valid,y_valid,X_test,y_test=(train_trans[:,:-1],
                                                           train_trans[:,-1],
                                                           valid_trans[:,:-1],
                                                           valid_trans[:,-1],
                                                           test_trans[:,:-1],
                                                           test_trans[:,-1])
            models={
                'XGBOOSTR':XGBRegressor(),
                'LinearRegression':LinearRegression(),
                'RDF':RandomForestRegressor(),
                'ADABOOST':AdaBoostRegressor(),
                'SVR':SVR(),
                'CatBoostRegressor':CatBoostRegressor(verbose=0),
                'KNB':KNeighborsRegressor(),
                'DTR':DecisionTreeRegressor()
                }
            df=train_evaluate_model(X_train=X_train,y_train=y_train,X_valid=X_valid,
                                 y_valid=y_valid,
                                 X_test=X_test,y_test=y_test,models=models)
            
            #best model
            best_model_name=df.index[0]
            best_model_score=df.iloc[0,-1]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model selected:{best_model_name}")

            #save model
            save_object(self.model_trainer_config.model_object_file,
                        obj=best_model
                        )

            return best_model_name,best_model_score


        
            
        except Exception as e:
            raise CustomException(e)