import random as rnd
import math

def normalize(input, max_value):
    normalized = []
    
    for i in input:
        normalized.append(round(i / max_value, 2))
    return normalized
    

def split_training_data(data, percentage_val_data):
    
    if percentage_val_data < 0.1 or percentage_val_data > 0.5:
        print('Percentage must be between 0.5 and 0.9!')
        return [], []

    n = math.ceil(len(data) * percentage_val_data)
    test_data = data
    val_data = []
    
    for i in range(n):
        val_data.append(test_data.pop(rnd.randrange(0,len(test_data))))
    
    return test_data, val_data