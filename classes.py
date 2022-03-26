import imp
import math
from os import stat
import random as rnd
from re import T
import numpy as np
import helpers as hlp
import pandas as pd


class Neuron:
    def __init__(self, id):
        self.id = id
        self.output = 0 if self.id[1] > 0 else 1 # bias neurons have id 0 and get an output of 1
    
    def propagate(self, prev_layer, current_layer):
        # input neurons and bias neurons should not propagate
        if (self.id[0] == 0 or self.id[1] == 0):
            return

        net_input = 0
        prev_neurons = []
        prev_neurons = prev_layer.neurons.copy()
        prev_neurons.insert(0,current_layer.bias_neuron)

        for n in prev_neurons:
            try:
                net_input += n.output * current_layer.weights[n.id[1]][self.id[1] - 1]
            except:
                print('Error i = ' + str(self.id) + ' j = ' + str(n.id) + ' w = ' + str(current_layer.weights[n.id[1]][self.id[1] - 1]))

        self.output = round(self.activate(net_input, current_layer.activation), 2)

    def activate(self, input, activation):
        if (self.id[1] == 0):
            return 1

        match activation:
            case 'identity':
                return input
            case 'relu':
                return max(0, input)
            case 'binary_step':
                return 0 if input < 0 else 1
            case 'sigmoid':
                return 1 / (1 + math.exp(-input))
            case 'tanh':
                return math.tanh(input)


class Layer:
    def __init__(self, id, num_of_neurons, activation = 'identity', prev_layer = None):
        self.id = id
        self.num_of_neurons = num_of_neurons
        self.activation = activation if activation in ['identity', 'relu', 'binary_step', 'sigmoid', 'tanh'] else 'identity'
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
                    if i == 0:
                        self.weights[i][j] = -round(rnd.random(), 2) # negative weights for bias neurons
                    else:
                        self.weights[i][j] = round(rnd.random(), 2) # positive weights for normal neurons
        else:
            self.weights = []

    def add_neuron(self, output):
        self.neurons.append(Neuron((self.id, len(self.neurons + 1))))

    def get_weights(self):
        return pd.DataFrame(self.weights)


class Network_Model:
    def __init__(self, id = 0):
        self.id = id
        self.layers = []

    def add_layer(self, num_of_neurons, activation = 'identity'):
        if num_of_neurons < 1:
            print('Number of neurons has to be larger or equal to 0!')
            return
        
        if activation not in ['identity', 'relu', 'binary_step', 'sigmoid', 'tanh']:
            print(f'Activation function "{activation}" does not exist!')
            return

        if len(self.layers) > 0:
            self.layers.append(Layer(len(self.layers), num_of_neurons, activation, self.layers[-1]))
        else:
            self.layers.append(Layer(len(self.layers), num_of_neurons, activation))
    
    def plot_network(self):
        for l in self.layers:
            s = ''
            for n in l.neurons:
                s += '<div style="border-style:outset; border-radius: 1ex; border-color: white; padding: 0.5ex; text-align: center; float: left; margin: 0.25ex; width: fit-content">'+ str(n.id) + '<br>output ' + str(n.output) + '</div>'
            hlp.printmd(s)

    def train(self, train_df_x, train_df_y, val_df_x, val_df_y, mode = 'online', epochs = 10):
        # ToDo: Validate training data (is numeric? is tuple? is not empty?
        
        if mode not in ['online', 'offline']:
            print('Invalid mode given. Must be "online" or "offline".')
            return

        # Training
        train_data_x = train_df_x.values.tolist()
        train_data_y = train_df_y.values.tolist()

        # x ... input vector with components x_i
        # y ... output vector with components y_i
        # p ... training sample with components p_i
        # t ... teaching input with components t_i
        # E_p ... Error vector for training sample p

        for e in range(1, epochs + 1):
            E_e = 0 # Cumulative error for the epoch
            for i,p in enumerate(train_data_x):
                # output vector
                y = self.predict(p) 

                # specific error
                t = train_data_y[i]
                E_p = 0 # Error for a specific training set
                temp_sum = 0
                for j,y_j in enumerate(y):
                    temp_1 = (y_j - (t[j] if type(t) == list else t))
                    temp_sum += temp_1 * temp_1
                E_p = 1/2 * temp_sum
                E_e += E_p

                # backpropagate
                layers = self.layers.copy()
                layers.reverse()

    def predict(self, input: list):
        # write input to first layer
        if len(input) != len(self.layers[0].neurons):
            print('Number of input values does not match number of input neurons!')
            return

        # write input to layer 0 neurons
        for i, n in enumerate(self.layers[0].neurons):
            n.output = input[i]
        
        # propagate
        for l in self.layers:
            if l.id != 0:
                for n in l.neurons:
                    n.propagate(self.layers[l.id - 1], l)
    
        output = []
        for n in self.layers[-1].neurons:
            output.append(n.output)
        
        return output


