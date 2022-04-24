import random as rnd
import networkx as nx
import pandas
import plotly.graph_objects as go
from model import helpers as hlp
from model.classes.layer import Layer

class Network:
    # constructor
    def __init__(self):
        '''Creates a network model object.'''
        self.layers = []


    # main functions
    def add_layer(self, num_of_neurons: int, propagation_function = 'weighted_sum', activation_function = 'identity', fixed_weight = None):
        '''Adds a new layer to the model and fills it with neurons.

        Parameters
        ----------
            num_of_neurons (int): Number of neurons for the layer.
            propagation_function (string): Type of propagation function that neurons of the layer use. Options are `'weighted_sum'` (default `'weighted_sum'`).
            activation_function (string): Type of propagation function that neurons of the layer use. Options are `'identity'`, `'relu'`, `'sigmoid'`, `'tanh'` (default `'identity'`)
            fixed_weight (float): sets the value of all weights if the value is not `None` (default `None`)
        '''
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
            self.layers.append(Layer(len(self.layers), num_of_neurons, propagation_function, activation_function, fixed_weight=fixed_weight, prev_layer=self.layers[-1]))
        else:
            self.layers.append(Layer(len(self.layers), num_of_neurons, propagation_function, activation_function, fixed_weight=fixed_weight))

    def update(self):
        '''Makes each neuron of the layer run it's update function.'''
        for l in self.layers:
            if l.id == 0: continue
            prev_neurons = self.layers[l.id - 1].neurons.copy()
            prev_neurons.insert(0, l.bias_neuron) # add bias neuron to front of list
            l.update(prev_neurons)

    def predict(self, input: list):
        '''Computes a model output based on a given input.
        
        Parameters
        ----------
            input (list): Vector (list) of input values for which a prediction is to be made.
        
        Returns
        ----------
            prediction (list): Vector (list) of predictions based on the input vector.
        '''
        # write input to first layer
        if len(input) != len(self.layers[0].neurons):
            print('Number of input values does not match number of input neurons!')
            return

        # write input to layer 0 neurons
        for i, n in enumerate(self.layers[0].neurons):
            n.output = input[i]
        
        self.update()

        prediction = []
        for n in self.layers[-1].neurons:
            prediction.append(n.output)
        
        return prediction


    # other functions
    def create_validation_list(self, val_df_x, val_df_y):
        '''Creates a list used to validate the predictions of a model using a validation dataset as input.
        
        Parameters
        ----------
            val_df_x (pandas.Dataframe): Validation input values
            val_df_y (pandas.Dataframe): Desired outcomes
        
        Returns
        ----------
            val_list (list): List of tuples of the form (prediction by the model, desired outcome)
        '''
        val_data_x = val_df_x.values.tolist()
        val_data_y = val_df_y.values.tolist()
        val_list = []

        for i,v in enumerate(val_data_x):
            val_list.append((self.predict(v), val_data_y[i]))
            
        return val_list

    def validate_binary(self, df_val_x_norm, df_val_y, debug = False):
        '''Validates the quality of a model based on a normalized validation set.
        
        Parameters
        ----------
            df_val_x_norm (pandas.Dataframe): Normalized validation input values
            df_val_y (pandas.Dataframe): Desired outcomes
            debug (bool): If `True`, validation results are logged in detail (default `False`)
        
        Returns
        ----------
            val_list (list): List of tuples of the form (prediction by the model, desired outcome)
        '''
        val_list = self.create_validation_list(df_val_x_norm, df_val_y)
        misclassifications = 0

        if len(val_list[0][0]) == 1 and len(val_list[0][1]):
            for v in val_list:
                if len(v[0]) == 1 and len(v[1]):
                    p = 1.0 if v[0][0] > 0.5 else 0.0
                    if p != v[1][0]:
                        misclassifications += 1
                        if debug: print(f'pred {v[0][0]}={p}, correct {v[1]} => X')
                    else:
                        if debug: print(f'pred {v[0][0]}={p}, correct {v[1][0]} => OK')

            print(f'{misclassifications} of {len(val_list)} misclassified.')
        else:
            print('Not implemented yet.')
            return []

        return val_list

    # visualization
    def plot_network(self, show_neurons = True, show_weights = True):
        '''Visualizes the network using NetworkX and Plotly. Neurons are plotted as Nodes, weights are plotted as lines. A neuron's id is shown. When hovering, additional information is shown.
        
        Parameters
        ----------
            show_nodes (bool): If `True`, neurons are visualized (default `True`)
            show_edges (bool): If `True`, weights are visualized (default `True`)
        '''
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

        if (not show_neurons and not show_weights): return

        if (show_weights):
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

        if (show_neurons):
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

        if (show_weights):
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            data.append(edge_trace)

        if (show_neurons):
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


class Feed_Forward(Network):
    # main functions
    def train(self,
        train_df_x: pandas.DataFrame,
        train_df_y: pandas.DataFrame,
        mode: str = 'online',
        epochs: int = 100,
        max_error = 0.0,
        adaptive_learning_rate: bool = False,
        min_learning_rate = 0.0,
        default_learning_rate = 0.5,
        momentum_factor = 0.0,
        weight_decay_factor = 0.0,
        shuffle: bool = False,
        debug: bool = False):
        '''Trains the model using normalized training data.
        
        Parameters
        ----------
            train_df_x (pandas.Dataframe): Training set.
            train_df_y (pandas.Dataframe): Training input.
            mode (string): Error evaluating method to be used. Options are `'online'`, `'offline'` (default `'online'`).
            epochs (int): Number of training iterations (default `100`).
            max_error (float): Maximum average error per epoch. If the average error falls below this threshhold, the training is aborted (default `0.0`).
            adaptive_learning_rate (bool): If `True`, the learning rate is adapted during the training process depending on the error dynamic (default `False`).
            min_learning_rate (float): Minimum learning rate the model should use. If the learning rate falls below this threshhold, the training is aborted (default `0.0`).
            default_learning_rate (float): Default learning rate the model uses to train (default `0.5`).
            momentum_factor (float): Factor the model uses for momentum learning. If the value is 0, momentum learning is not used. The deal value should be between `0.6` and `0.9` (default `0.0`).
            weight_decay_factor (float): Factor the model uses for weight decay. If the value is 0, momentum learning is not used. The deal value should be between `0.001` and `0.02` (default `0.0`).
            shuffle (bool): If `True`, the training set is shuffeled for each epoch (default `False`)
            debug (bool): If `True`, detailed information is logged during the training process (default `False`).
        
        Returns:
        ----------
            history (list): List of errors for each epoch.
        '''
        if (mode not in ['online', 'offline']): return []

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

            if epoch > 1 and shuffle:
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
                Err_p = 0.5 * sum([k*l for k,l in zip(E_p,E_p)])
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

                        h.delta = delta_h

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

            # add average Error to list of average Errors
            Err.append(Err_e_avg)

            # adapt learning rate based on error
            error_dynamic = 0

            if adaptive_learning_rate:
                error_dynamic = hlp.get_error_dynamic(Err, 10)
                if  error_dynamic > 0:
                    learning_rate /= 2

            if debug:
                print(f'epoch = {epoch} | average error = {Err_e_avg} | error dynamic = {error_dynamic} | learning rate = {learning_rate}')

            if Err[-1] < max_error:
                print(f'Training finished. Max error rate of < {max_error} reached on epoch {epoch}.')
                break

            if learning_rate < min_learning_rate:
                print(f'Training aborted. Learning rate reached < {min_learning_rate} on epoch {epoch}.')
                break
        
            if epoch == epochs:
                print(f'Training finished. Number of epochs reached.')

        return Err