import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

import pickle as pk
import dill

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=Path(file_path).parent
        dir_path.mkdir(parents=True,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pk.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e)

