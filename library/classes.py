import imp
import math
from os import stat
import random as rnd
from re import T
import numpy as np
from library import helpers as hlp
import pandas as pd


class Neuron:
    def __init__(self, id):
        self.id = id

        self.net = 0
        self.activation = 0
        self.output = 0 if self.id[1] > 0 else 1 # bias neurons have id 0 and get an output of 1

        self.delta = 0
    
    def do_update(self, prev_layer, current_layer):
        """Updates a neurons output value by propagating and activating."""
        # input neurons and bias neurons should not propagate
        if (self.id[0] == 0 or self.id[1] == 0): return False

        if not self.do_propagate(prev_layer, current_layer): return False
        if not self.do_activate(current_layer.activation_function): return False
        if not self.do_output(): return False

    def do_propagate(self, prev_layer, current_layer):
        """Propagates and computes the net input."""
        try:
            net_input = 0
            prev_neurons = []
            prev_neurons = prev_layer.neurons.copy()
            prev_neurons.insert(0,current_layer.bias_neuron) # add bias neuron to front of list

            for n in prev_neurons:
                net_input += n.output * current_layer.weights[n.id[1]][self.id[1] - 1]

            self.net = net_input
            return True
        except:
            return False

    def do_activate(self, activation):
        """Computes the activation value using the net input and the layers activation function."""
        if (self.id[1] == 0):
            return False

        try:
            match activation:
                case 'identity':
                    self.activation = hlp.activate_identity(self.net)
                    return True
                case 'relu':
                    self.activation = hlp.activate_relu(self.net)
                    return True
                case 'sigmoid':
                    self.activation = hlp.activate_sigmoid(self.net)
                    return True 
                case 'tanh':
                    self.activation = hlp.activate_tanh(self.net)
                    return True
        except:
            return False

    def do_output(self):
        """Computes the neuron's output using it's activation value."""
        if (self.id[1] == 0):
            return False

        try:
            self.output = self.activation
        except:
            return False

    def do_activate_der(self, activation_function):
        try:
            match activation_function:
                case 'identity':
                    return hlp.activate_identity_der(self.net)
                case 'relu':
                    return hlp.activate_relu_der(self.net)
                case 'sigmoid':
                    return hlp.activate_sigmoid_der(self.net) 
                case 'tanh':
                    return hlp.activate_tanh_der(self.net)
        except:
            return False

    def set_delta(self, delta):
        """Sets the delta value for a neuron."""
        self.delta = delta


class Layer:
    def __init__(self, id, num_of_neurons, activation_function = 'identity', prev_layer = None):
        self.id = id
        self.num_of_neurons = num_of_neurons
        self.activation_function = activation_function if activation_function in ['identity', 'relu', 'sigmoid', 'tanh'] else 'identity'
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
                    self.weights[i][j] = (rnd.random() - 0.5) * 2
        else:
            self.weights = []

    def add_neuron(self, output):
        """Adds a neuron to a layer."""
        self.neurons.append(Neuron((self.id, len(self.neurons + 1))))

    def get_weights(self):
        """Returns the weights of a layer as a dataframe."""
        return pd.DataFrame(self.weights)


class Network_Model:
    def __init__(self, id = 0):
        self.id = id
        self.layers = []

    def add_layer(self, num_of_neurons, activation_function = 'identity'):
        """Adds a new layer to the model and fills it with neurons."""
        if num_of_neurons < 1:
            print('Number of neurons has to be larger or equal to 0!')
            return
        
        if activation_function not in ['identity', 'relu', 'sigmoid', 'tanh']:
            print(f'Activation function "{activation_function}" does not exist!')
            return

        if len(self.layers) > 0:
            self.layers.append(Layer(len(self.layers), num_of_neurons, activation_function, self.layers[-1]))
        else:
            self.layers.append(Layer(len(self.layers), num_of_neurons, activation_function))
    
    def plot_network(self):
        """Visualizes layers and neurons."""
        for l in self.layers:
            s = ''
            for n in l.neurons:
                s += '<div style="border-style:outset; border-radius: 1ex; border-color: white; padding: 0.5ex; text-align: center; float: left; margin: 0.25ex; width: fit-content">'+ str(n.id) + '<br>net ' + str(n.net) + '<br>act ' + str(n.activation) + '<br>out ' + str(n.output) + '</div>'
            hlp.printmd(s)

    def train(self, train_df_x, train_df_y, mode = 'online', epochs = 100, default_learning_rate = 0.5, adaptive_learning_rate = False, shuffle = False, debug = False):
        """Trains the model using normalized training data."""
        train_data_x_orig = train_df_x.values.tolist()
        train_data_y = train_df_y.values.tolist()

        learning_rate = default_learning_rate

        #region iterate over epochs
        Err = []
        for epoch in range(1, epochs + 1):
            Err_e = 0

            #region shuffle training set
            train_data_x = train_data_x_orig.copy()

            if shuffle:
                train_data_x = train_data_x_orig.copy()
                rnd.shuffle(train_data_x)

            #endregion

            #region iterate of all training sets
            for i,p in enumerate(train_data_x):
                # output vector
                y = self.predict(p)

                # specific error
                t = train_data_y[train_data_x_orig.index(p)]
                Err_p = 0 # root mean square error
                E_p = [] #error vector
                temp_sum = 0

                for j,y_j in enumerate(y):
                    E_p.append((t[j] if type(t) == list else t) - y_j)
                temp_sum = 0
                for e in E_p:
                    temp_sum += e * e
                Err_p = 1/2 * temp_sum

                Err_e += Err_p

                # backpropagate
                delta_w = []

                for layer in range(len(self.layers) - 1, 0, -1):
                    is_output_layer = layer == len(self.layers) - 1

                    # current layer neurons
                    neurons_h = self.layers[layer].neurons

                    # previous layer neurons
                    neurons_k = self.layers[layer - 1].neurons.copy()
                    neurons_k.insert(0, self.layers[layer].bias_neuron)

                    # following layer neurons, weights
                    neurons_l = None if is_output_layer else self.layers[layer + 1].neurons
                    weights_l = None if is_output_layer else self.layers[layer + 1].weights
                    
                    act_func = self.layers[layer].activation_function
                    delta_w.insert(0,[])

                    for h in neurons_h:
                        delta_w[0].append([])
                        act_der = h.do_activate_der(act_func)
                        for k in neurons_k:
                            del_h = 0

                            # del_h for output neurons
                            if is_output_layer:
                                del_h = act_der * E_p[h.id[1] - 1]

                            # del_h for hidden neurons
                            else:
                                for l in neurons_l:
                                    del_h += l.delta * weights_l[h.id[1]][l.id[1] - 1]
                                del_h *= act_der
                                
                            h.set_delta(del_h)
                            w = learning_rate * k.output * del_h
                            delta_w[0][-1].append(w)

                # update weights
                for i,l in enumerate(self.layers):
                    if i != 0:
                        delta_i_t = list(map(list, zip(*delta_w[i - 1])))
                        for k in range(len(l.weights)):
                            for h in range(len(l.weights[0])):
                                l.weights[k][h] = l.weights[k][h] + delta_i_t[k][h]
            #endregion

            Err_e = 1 / len(train_data_x) * Err_e #average Error of all training sets per epoch
            Err.append(Err_e)

            if adaptive_learning_rate:
                if len(Err) > 1:
                    if Err[-1] > Err[-2]:
                        learning_rate *= 0.7
                    else:
                        learning_rate *= 1.3
        
            if debug: print(f'epoch = {epoch} | error = {Err_e} | learning rate = {learning_rate}')
        #endregion
        
        return Err

    def predict(self, input: list):
        """Computes a model output based on a given input."""
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
                    n.do_update(self.layers[l.id - 1], l)
    
        output = []
        for n in self.layers[-1].neurons:
            output.append(n.output)
        
        return output
