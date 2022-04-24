import random as rnd
import math
from IPython.display import Markdown, display
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# data preprocessing
def normalize(dataframe: pd.DataFrame, min_val, max_val, lower_bound = -1.0, upper_bound = 1.0):
    '''Normalizes a the values of a dataframe within a given boundary to a given range.

    Parameters
    ----------
        df_orig (pandas.Dataframe): Data that is to be normalized.
        min_val (float): Lowest possible value of a datapoint.
        max_val (float): Highest possible value of a datapoint.
        lower_bound (float): Lower end of the range of normalized values (default `-1`)
        upper_bound (float): Lower end of the range of normalized values (default `1`)

    Returns
    ----------
        df_norm (pandas.Dataframe): Normalized data
    
    References
    ----------
        Normalization: https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network#:~:text=Batch%20Normalization,convergence%20of%20the%20training%20process.
    '''
    if type(dataframe) != pd.DataFrame:
        print('Data must be of type pandas.Dataframe')
        return

    df_not_norm = dataframe.copy()

    data_not_norm = df_not_norm.values.tolist()
    data_transposed = list(map(list, zip(*data_not_norm)))

    for i, column in enumerate(data_transposed):
        for j, value in enumerate(column):
            data_transposed[i][j] = (value - min_val) / (max_val - min_val) * (upper_bound - lower_bound) + lower_bound

    data_norm= list(map(list, zip(*data_transposed)))
    df_norm = pd.DataFrame(data_norm, columns = df_not_norm.columns)
    return df_norm

def split_training_val_data(dataframe: pd.DataFrame, percentage_val_data = 0.2):
    '''Splits the rows of a given dataframe into two new dataframes randomly using a percentage.
    
    Parameters
    ----------
        dataframe (pandas.Dataframe): Dataset that is to be split
        percentage_val_data (float): Percentage of data that is to be split from the given data (default `0.2`)

    Returns
    ----------
        test_df (pandas.Dataframe): Test data
        val_df (pandas.Dataframe): Validation data
    '''
    val_df = dataframe.sample(frac=percentage_val_data)
    test_df = pd.concat([dataframe,val_df]).drop_duplicates(keep=False)
    
    return test_df, val_df

def split_input_output_data(dataframe: pd.DataFrame, x_cols: list[str], y_cols: list[str]):
    '''Splits a given dataframe into two new dataframes using the respective columns.
    
    Parameters
    ----------
        dataframe (pandas.Dataframe): Dataset that is to be split
        x_cols (list[string]): Column headers of input values
        y_cols (list[string]): Columns headers of output values

    Returns
    ----------
        x_df (pandas.Dataframe): Input data
        y_df (pandas.Dataframe): Output data
    '''
    x_df = dataframe.copy()
    y_df = x_df[y_cols].copy()
    x_df = x_df.drop(y_cols, axis=1)

    return x_df, y_df


# activation functions
def activate_identity(value):
    '''Computes the result for a given value using the identity function.
    
    Parameters
    ----------
        value (float): input for which the activation function is to be computed.
        
    Returns
    ----------
        result (float): Result of the calculation
    '''
    return value

def activate_relu(value):
    '''Computes the result for a given value using the relu function.
    
    Parametes
    ----------
        value (float): input for which the activation function is to be computed.
        
    Returns
    ----------
        result (float): Result of the calculation
    '''
    return max(0.0, value)
    
def activate_sigmoid(value):
    '''Computes the result for a given value using the sigmoid function.
    
    Parameters
    ----------
        value (float): input for which the activation function is to be computed.
        
    Returns
    ----------
        result (float): Result of the calculation
    '''
    return 1.0 / (1.0 + math.exp(-value))

def activate_tanh(value):
    '''Computes the result for a given value using the hyperbolic tangent function.
    
    Parameters
    ----------
        value (float): input for which the activation function is to be computed.
    
    Returns
    ----------
        result (float): Result of the calculation
    '''
    return math.tanh(value)


# derivations of activation functions
def activate_identity_der(value):
    '''Computes the result for a given value using the first derivative of the identity function. A flatpoint elimination constant of 0.1 is used.
    
    Parameters
    ----------
        value (float): input for which the first derivation of the activation function is to be computed.
    
    Returns
    ----------
        result (float): Result of the calculation
    '''
    return 1.0

def activate_relu_der(value):
    '''Computes the result for a given value using the first derivative of the relu function. It is technically not defined at 0, but will return 1. A flatpoint elimination constant of 0.1 is used.
    
    Parameters
    ----------
        value (float): input for which the first derivation of the activation function is to be computed.
    
    Returns
    ----------
        result (float): Result of the calculation
    '''
    return 0.0 + 0.1 if value < 0.0 else 1.0
    
def activate_sigmoid_der(value):
    '''Computes the result for a given value using the first derivative of the sigmoid function. A flatpoint elimination constant of 0.1 is used.
    
    Parameters
    ----------
        value (float): input for which the first derivation of the activation function is to be computed.
    
    Returns
    ----------
        result (float): Result of the calculation
    
    References
    ----------
        Sigmoid derivative https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    '''
    sigm = 1.0 / (1.0 + math.exp(-value))
    return sigm * (1.0 - sigm) + 0.1

def activate_tanh_der(value):
    '''Computes the result for a given value using the first derivative of the hyperbolic tangent function. A flatpoint elimination constant of 0.1 is used.
    
    Parameters
    ----------
        value (float): input for which the first derivation of the activation function is to be computed.

    Returns
    ----------
        result (float): Result of the calculation
    
    References
    ----------
        Hyperbolic tangent derivative https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    '''
    return 1.0 - math.tanh(value) *  math.tanh(value) + 0.1


# visualization
def printmd(string):
    '''Prints a string as markup.
    
    Parameters
    ----------
        string (string): Text that is to be displayed as markup. HTML/CSS can be used here.
    '''
    display(Markdown(string))

def plot_learning_curve(data, xlabel = 'Epoch', ylabel = 'Average Error', title = 'Learning Curve'):
    '''Uses an error history generated by a neural network training method the to plot the learning curve.

    Parameters
    ----------
        data (list[float]): Data that is to be plotted
        xlabel (string): Label for the x-axis of the plot (default `'Epoch'`)
        ylabel (string): Label for the y-axis of the plot (default `'Average Error'`)
        title (string): Title of the plot (default `'Learning Curve'`)
    '''
    epoch = list(range(1, len(data) + 1))
    df = pd.DataFrame({xlabel : epoch, ylabel : data})
    fig = px.line(
        df,
        x=xlabel,
        y=ylabel,
        title=title
    )
    fig.show()


# other functions
def get_error_dynamic(error_list, dynamic_range = 10):
    '''Computes the dynamic error value based on a given list of error values and a desired range.

    Parameters
    ----------
        error_list (list[float]): List of computed error values
        dynamic_range (int): number of most recent error values that are to be considered for the calculation

    Returns
    ----------
        dynamic (float): value for the error dynamic. Is 0 if the given error list is too small.
    '''
    if len(error_list) < dynamic_range + 1:
        return 0
    dynamic = 0.0
    for i in range(len(error_list) - dynamic_range, len(error_list)):
        dynamic += (error_list[i] - error_list[i - 1]) / abs(error_list[i - 1])
    dynamic /= dynamic_range
    return dynamic