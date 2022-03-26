from cmath import inf
import random as rnd
import math
from IPython.display import Markdown, display
import pandas as pd

def printmd(string):
    display(Markdown(string)) 

def normalize(df_orig: pd.DataFrame, min_val, max_val, lower_bound = -1, upper_bound = 1):

    if type(df_orig) != pd.DataFrame:
        print('Data must be of type Pandas.Dataframe')
        return

    df_not_norm = df_orig.copy()

    data_not_norm = df_not_norm.values.tolist()
    data_transposed = list(map(list, zip(*data_not_norm)))

    for i, column in enumerate(data_transposed):
        for j, value in enumerate(column):
            # https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network#:~:text=Batch%20Normalization,convergence%20of%20the%20training%20process.
            data_transposed[i][j] = (value - min_val) / (max_val - min_val) * (upper_bound - lower_bound) + lower_bound

    data_norm= list(map(list, zip(*data_transposed)))
    df_norm = pd.DataFrame(data_norm, columns = df_not_norm.columns)
    return df_norm


def split_training_val_data(df: pd.DataFrame, percentage_val_data = 0.2):

    val_df = df.sample(frac=percentage_val_data)
    test_df = pd.concat([df,val_df]).drop_duplicates(keep=False)
    
    return test_df, val_df

def split_input_output_data(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]):
    x_df = df.copy()
    y_df = x_df[y_cols].copy()
    x_df = x_df.drop(y_cols, axis=1)

    return x_df, y_df