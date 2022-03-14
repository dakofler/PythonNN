import math
from os import stat
import random as rnd
import numpy as np
import helpers as hlp


class Neuron:
    def __init__(self, id):
        self.id = id
        self.output = 0
        self.bias = 0 if self.id[0] == 0 else round(rnd.random(), 2) # if layer id is 0, it is an input layer
    
    def propagate(self, prev_layer_neurons, current_layer_weights, current_layer_activation_function):
        net_input = 0

        for n in prev_layer_neurons:
            net_input += n.output * current_layer_weights[n.id[1]][self.id[1]]
            
        net_input -= self.bias
        self.output = round(self.activate(net_input, current_layer_activation_function), 2) # ToDo: How does the previous activation play a role?

    def activate(self, input, activation):
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

        for i in range(self.num_of_neurons):
            self.neurons.append(Neuron((self.id, i), ))
        
        if self.id > 0:
            self.weights = np.zeros((prev_layer.num_of_neurons, self.num_of_neurons))
            
            for i in range(len(self.weights)):
                for j in range(len(self.weights[0])):
                    self.weights[i][j] = round(rnd.random(), 2)

    def add_neuron(self):
        self.neurons.append(Neuron((self.id, len(self.neurons))))


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
            s = '|'
            for n in l.neurons:
                s += ' ' + str(n.id) + ', o = ' + str(n.output) + ', b = ' + str(n.bias) + ' |'
            print(s)
    
    def train(self, training_data, mode = 'online', epochs = 10):
        # validation

        # ToDo: Verify training data (is numeric? is tuple? is not empty?
        
        if mode not in ['online', 'offline']:
            print('Invalid mode given. Must be "online" or "offline".')
            return

        if epochs < 1 or epochs > 100: epochs = 10

        # split training data into training and validation data 70-30
        # ToDo

        # propagate
        # ToDo

        # error vector
        # ToDo

        # backpropagate
        # ToDo




    def predict(self, input):
        # write input to first layer
        if len(input) != len(self.layers[0].neurons):
            print('Number of input values does not match number of input neurons!')
            return

        for i, n in enumerate(self.layers[0].neurons):
            n.output = input[i]
        
        # propagate
        for l in self.layers:
            if l.id == 0: continue # prevent input neurons from propagating

            for n in l.neurons:
                n.propagate(self.layers[l.id - 1].neurons, l.weights, l.activation)
    
        for n in self.layers[-1].neurons:
            print(n.output)


