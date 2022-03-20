import math
from os import stat
import random as rnd
from re import T
import numpy as np
import helpers as hlp


class Neuron:
    def __init__(self, id):
        self.id = id
        self.output = 0
        self.bias = 0 if self.id[0] == 0 else round((rnd.random() - 0.5) * 2, 2) # if layer id is 0, it is an input layer
    
    def propagate(self, prev_layer_neurons, current_layer_weights, current_layer_activation_function):
        net_input = 0

        for n in prev_layer_neurons:
            net_input += n.output * current_layer_weights[n.id[1]][self.id[1]]
            
        net_input -= self.bias
        self.output = round(self.activate(net_input, current_layer_activation_function), 2)

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

        # add neurons at initialization
        for i in range(self.num_of_neurons):
            self.neurons.append(Neuron((self.id, i), ))
        
        # if hidden or output layer, add weights
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
    
    def plot_network_pretty(self):
        for l in self.layers:
            s = ''
            for n in l.neurons:
                s += '<div style="border-style:outset; border-radius: 1ex; border-color: white; padding: 0.5ex; text-align: center; float: left; margin: 0.25ex; width: fit-content">'+ str(n.id) + '<br>output ' + str(n.output) + '<br>bias ' + str(n.bias) + '</div>'
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

        for i,p in enumerate(train_data_x):
            print('p: ' + str(p))

            # output vector
            y = self.predict(p)
            print('y: ' + str(y))

            # teaching input
            t = train_data_y[i]
            print('t: ' + str(t))

            # error vector
            E_p = []
            for j,y_i in enumerate(y):
                E_p.append((t[j] if type(t) == list else t) - y_i)
            print('E_p:' + str(E_p))

            # specific error
            sum_square = 0
            for e in E_p:
                sum_square += e * e
            Err_p = 1/2 * sum_square
            print('Err_p:' + str(Err_p))
            print('')

            # backpropagate
            # ToDo

    def predict(self, input: list):
        # write input to first layer
        if len(input) != len(self.layers[0].neurons):
            print('Number of input values does not match number of input neurons!')
            return

        for i, n in enumerate(self.layers[0].neurons):
            n.output = input[i]
        
        # propagate
        for l in self.layers:
            if l.id == 0:
                continue # prevent input neurons from propagating

            for n in l.neurons:
                n.propagate(self.layers[l.id - 1].neurons, l.weights, l.activation)
    
        output = []
        for n in self.layers[-1].neurons:
            output.append(n.output)
        
        return output


