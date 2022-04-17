import random as rnd
from re import T
import numpy as np
from model import helpers as hlp
import pandas as pd
import networkx as nx
import plotly.graph_objects as go


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

        net_input = 0
        prev_neurons = []
        prev_neurons = prev_layer.neurons.copy()

        if current_layer.propagation_function == 'weighted_sum':
            prev_neurons.insert(0,current_layer.bias_neuron) # add bias neuron to front of list

            for n in prev_neurons:
                net_input += n.output * current_layer.weights[n.id[1]][self.id[1] - 1]

            self.net = net_input
            return True
        else:
            return False

    def do_activate(self, activation):
        """Computes the activation value using the net input and the layers activation function."""
        if (self.id[1] == 0):
            return False

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

    def do_output(self):
        """Computes the neuron's output using it's activation value."""
        if (self.id[1] == 0):
            return False

        self.output = self.activation

    def do_activate_der(self, activation_function):
        """Computes the the value for the derivative of a neurons activation using the net input."""

        match activation_function:
            case 'identity':
                return hlp.activate_identity_der(self.net)
            case 'relu':
                return hlp.activate_relu_der(self.net)
            case 'sigmoid':
                return hlp.activate_sigmoid_der(self.net) 
            case 'tanh':
                return hlp.activate_tanh_der(self.net)

    def set_delta(self, delta):
        """Sets the delta value for a neuron."""

        self.delta = delta


class Layer:
    def __init__(self, id, num_of_neurons, propagation_function, activation_function, prev_layer=None, fixed_weight=None):
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
                        self.weights[i][j] = (rnd.random() - 0.5) * 2
        else:
            self.weights = []

    def add_neuron(self, output):
        """Adds a neuron to a layer."""
        self.neurons.append(Neuron((self.id, len(self.neurons + 1))))

    def get_weights(self):
        """Returns the weights of a layer as a dataframe."""
        return pd.DataFrame(self.weights)


class Neural_Network:
    def __init__(self, id=0):
        self.id = id
        self.layers = []

    def add_layer(self, num_of_neurons, propagation_function = 'weighted_sum', activation_function = 'identity', fixed_weight=None):
        """Adds a new layer to the model and fills it with neurons."""
        if num_of_neurons < 1:
            print('Number of neurons has to be larger or equal to 0!')
            return
        
        if propagation_function not in ['weighted_sum', 'radial_basis']:
            print(f'Propagation function "{propagation_function}" does not exist!')
            return

        if activation_function not in ['identity', 'relu', 'sigmoid', 'tanh']:
            print(f'Activation function "{activation_function}" does not exist!')
            return

        if len(self.layers) > 0:
            self.layers.append(Layer(len(self.layers), num_of_neurons, propagation_function, activation_function, self.layers[-1], fixed_weight))
        else:
            self.layers.append(Layer(len(self.layers), num_of_neurons, propagation_function, activation_function, fixed_weight=fixed_weight))

    def update(self):
        """Updates each neuron in the network."""
        for l in self.layers:
            if l.id != 0:
                for n in l.neurons:
                    n.do_update(self.layers[l.id - 1], l)

    def predict(self, input: list):
        """Computes a model output based on a given input."""

        # write input to first layer
        if len(input) != len(self.layers[0].neurons):
            print('Number of input values does not match number of input neurons!')
            return

        # write input to layer 0 neurons
        for i, n in enumerate(self.layers[0].neurons):
            n.output = input[i]
        
        self.update()

        output = []
        for n in self.layers[-1].neurons:
            output.append(n.output)
        
        return output

    def plot_network(self, show_nodes=True, show_edges=True):
        """Visualizes the network using NetworkX and Plotly"""

        # create list of all neurons
        neurons = []
        for l in self.layers:
            for n in l.neurons:
                neurons.append(n)

        # instanciate graph
        graph = nx.Graph()

        # add neurons as nodes
        for n in neurons:
            graph.add_node(n.id)

        #calculate node positions
        plot_width = 1500 # 200 + len(self.layers) * 250
        temp = [n.id[1] for n in neurons]
        plot_height = 200 + max(temp) * 50 - 10

        pos = {}
        d_x = plot_width / len(self.layers)
        d_y = [plot_height / len(l.neurons) for l in self.layers]

        for n in neurons:
            current_layer_id = n.id[0]
            current_neuron_id = n.id[1]
            x = d_x / 2 + current_layer_id * d_x
            y = plot_height - d_y[current_layer_id] / 2 - (current_neuron_id - 1) * d_y[current_layer_id]

            pos[n.id] = [x, y]
        
        # add edges
        for l in self.layers:
            if l.id != 0:
                for n in l.neurons:
                    for n_k in self.layers[l.id - 1].neurons:
                        graph.add_edge(n_k.id, n.id)

        if (not show_nodes and not show_edges): return

        if (show_edges):
            edge_x = []
            edge_y = []

            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]

                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

        if (show_nodes):
            node_x = []
            node_y = []
            node_hover_text = []
            node_text = []
            node_adjacencies = []

            for node, adjacencies in enumerate(graph.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))

            for i, node in enumerate(graph.nodes()):
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_hover_text.append(f'net={neurons[i].net}<br>activation={neurons[i].activation}<br>output={neurons[i].output}')

        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=plot_width,
            height=plot_height
        )

        data = []

        if (show_edges):
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            data.append(edge_trace)

        if (show_nodes):
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                text=node_text,
                textfont=dict(color='#000000'),
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_hover_text,
                marker=dict(
                    color='white',
                    size=45,
                    opacity=1,
                    line=dict(width=0.5, color='black')
                )
            )
            data.append(node_trace)
        
        fig = go.Figure(data=data, layout=layout)
        fig.show()


class Feed_Forward(Neural_Network):
    def train(self,
        train_df_x,
        train_df_y,
        mode = 'online',
        epochs = 100,
        default_learning_rate = 0.5, learning_rate_p = 1, learning_rate_n = 1,
        momentum_factor = 0,
        shuffle = False,
        debug = False):
        """Trains the model using normalized training data."""

        train_data_x_orig = train_df_x.values.tolist()
        train_data_y_orig = train_df_y.values.tolist()

        learning_rate = default_learning_rate

        # iterate over epochs
        Err = []
        Err_e = []
        for epoch in range(1, epochs + 1):
            Err_e.clear()

            # shuffle training set
            train_data_x = train_data_x_orig.copy()

            if shuffle:
                train_data_x = train_data_x_orig.copy()
                rnd.shuffle(train_data_x)

            # iterate of all training sets
            for i,p in enumerate(train_data_x):

                # output vector
                y = self.predict(p)

                # training input
                t = train_data_y_orig[train_data_x_orig.index(p)]
                
                # error vector
                E_p = [] 
                for j,y_j in enumerate(y):
                    E_p.append((t[j] if type(t) == list else t) - y_j)

                # specific error
                Err_p = 1/2 * sum([k*l for k,l in zip(E_p,E_p)])
                Err_e.append(Err_p)

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
                        delta_h = 0

                        # delta_h for output neurons
                        if is_output_layer:
                            delta_h = act_der * E_p[h.id[1] - 1]

                        # delta_h for hidden neurons
                        else:
                            delta_sum = 0
                            for l in neurons_l:
                                w_h_l = weights_l[h.id[1]][l.id[1] - 1]
                                delta_sum += (l.delta * w_h_l)
                            delta_h = act_der * delta_sum

                        h.set_delta(delta_h)

                        # compute delta w
                        for k in neurons_k:
                            w = 0
                            if i > 0:
                                w = learning_rate * k.output * delta_h + momentum_factor * delta_w_prev[layer - 1][h.id[1] - 1][k.id[1]]
                            else:
                                w = learning_rate * k.output * delta_h
                            delta_w[0][-1].append(w)

                # update weights
                for i,l in enumerate(self.layers):
                    if i != 0:
                        delta_i_t = list(map(list, zip(*delta_w[i - 1])))
                        for k in range(len(l.weights)):
                            for h in range(len(l.weights[0])):
                                l.weights[k][h] = l.weights[k][h] + delta_i_t[k][h]
                
                delta_w_prev = delta_w.copy()
            
            # average Error of all training sets per epoch
            Err_e_avg = sum(Err_e) / len(Err_e)

            # adapt learning rate based on error
            if len(Err) >= 1:
                if Err_e_avg > Err[-1]:
                    learning_rate *= learning_rate_n
                else:
                    learning_rate *= learning_rate_p

            Err.append(Err_e_avg)
            if debug: print(f'epoch = {epoch} | error = {Err_e_avg} | learning rate = {learning_rate}')
        
        return Err