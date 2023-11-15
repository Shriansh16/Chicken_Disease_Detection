import os
import sys
import pandas as pd
sys.path.insert(0, 'D:\Chicken_Disease_Classification\src')
from logger import logging
from exception import CustomException
from utils import *
from components.prepering_base_model import *
from components.training import *



if __name__=='__main__':
    config2 = ConfigurationManager2()
    training_config = config2.get_training_config()
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train()