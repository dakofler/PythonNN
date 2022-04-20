import random as rnd
import numpy as np
import pandas as pd

from model.classes.neuron import Neuron

class Layer:
    def __init__(self, id, num_of_neurons, propagation_function, activation_function, prev_layer = None, fixed_weight = None):
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
        if self.id > 0:
            self.weights = np.zeros((prev_layer.num_of_neurons + 1, self.num_of_neurons))
            
            for i in range(len(self.weights)):
                for j in range(len(self.weights[0])):
                    if fixed_weight is not None:
                        self.weights[i][j] = fixed_weight
                    else:
                        self.weights[i][j] = (rnd.random() - 0.5) # initialize weights in the interal [-0.5, 0.5]
        else:
            self.weights = []

    def add_neuron(self, output):
        """Adds a neuron to a layer."""
        self.neurons.append(Neuron((self.id, len(self.neurons + 1))))

    def get_weights(self):
        """Returns the weights of a layer as a dataframe."""
        if self.id == 0 :
            print('The input layer does not have weights.')
            return
        return pd.DataFrame(self.weights)
