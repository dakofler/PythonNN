import random as rnd
import math
from IPython.display import Markdown, display
import pandas as pd

def printmd(string):
    display(Markdown(string))
    

def normalize(input, max_value):
    normalized = []
    
    for i in input:
        normalized.append(round(i / max_value, 2))
    return normalized
    

def split_training_data(data: list, percentage_val_data):

    if percentage_val_data < 0.1 or percentage_val_data > 0.5:
        print('Percentage must be between 0.5 and 0.9!')
        return [], []

    n = math.ceil(len(data) * percentage_val_data)
    test_data = list(data) 
    val_data = []
    
    for i in range(n):
        val_data.append(test_data.pop(rnd.randrange(0,len(test_data))))
    
    train_data_x = []
    train_data_y = []
    val_data_x = []
    val_data_y = []

    for t in test_data:
        train_data_x.append(t[: 3])
        train_data_y.append(t[3 :])

    for t in val_data:
        val_data_x.append(t[: 3])
        val_data_y.append(t[3 :])

    return train_data_x, train_data_y, val_data_x, val_data_y