from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    try:
        ingestion=DataIngestion("notebook")
        df=ingestion.initiate_data_ingestion()
        train_set,valid_set,test_set=ingestion.data_split(df)

        transformer=DataTransformation(filename="train.csv")
        train_path="artifacts/train.csv"
        valid_path="artifacts/valid.csv"
        test_path="artifacts/test.csv"

        train_trans,valid_trans,test_trans=transformer.initiate_data_transformation(train_path,valid_path,test_path)
        trainer=ModelTrainer()
        print(trainer.train_model(train_trans,valid_trans,test_trans))


    except:
        pass
