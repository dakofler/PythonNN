import random as rnd
import numpy as np
import pandas as pd
from model.classes.neuron import Neuron

class Layer:
    # constructor
    def __init__(self, id: int, num_of_neurons: int, propagation_function: str, activation_function: str, fixed_weight, prev_layer = None):
        '''Creates a network model layer object.
        
        Parameters
        ----------
            id (int): ID of the layer.
            num_of_neurons (int): Number of neurons that the layer should contain.
            propagation_function (string): Propagation function that neurons of the layer use
            activation_function (string): Actviation function that neurons of the layer use
            fixed_weight (float | None): If `None`, weights are randomly initialized with a value between `-0.5` and `0.5` (default `None`)
            prev_layer (model.Layer): Previous Layer. Used for the initialization of weights (default `None`)
        '''
        self.id = id
        self.num_of_neurons = num_of_neurons
        self.activation_function = activation_function
        self.propagation_function = propagation_function
        self.neurons = []

        # add bias neuron
        if self.id > 0:
            self.bias_neuron = Neuron((self.id, 0))
        else:
            self.bias_neuron = None

        # add neurons at initialization
        for i in range(1, self.num_of_neurons + 1): # Neuron 0 is always the bias neuron
            self.neurons.append(Neuron((self.id, i)))
        
        # if hidden or output layer, add weights
        if prev_layer is not None:
            self.weights = np.zeros((prev_layer.num_of_neurons + 1, self.num_of_neurons))
            
            for i in range(len(self.weights)):
                for j in range(len(self.weights[0])):
                    if fixed_weight is not None:
                        self.weights[i][j] = fixed_weight
                    else:
                        self.weights[i][j] = (rnd.random() - 0.5) # initialize weights in the interal [-0.5, 0.5]
        else:
            self.weights = []

    # main functions
    def add_neuron(self):
        '''Adds a neuron to a layer.'''
        self.neurons.append(Neuron((self.id, len(self.neurons + 1))))

    def update(self, prev_layer_neurons):
        '''Makes each neuron of the layer run it's update function.
        
        Parameters
        ----------
            prev_layer_neurons (list): List of neurons of the previous layer.
        '''
        if self.id == 0: return
        for n in self.neurons:
            n.do_update(prev_layer_neurons, self)

    # other functions
    def get_weights(self):
        '''Returns the weights of a layer formatted as a dataframe.
        
        Returns
        ----------
            dataframe (pandas.Dataframe): Weights of the layer
        '''
        if self.id == 0 :
            print('The input layer does not have weights.')
            return
        return pd.DataFrame(self.weights)