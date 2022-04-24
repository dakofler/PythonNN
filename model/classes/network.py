import random as rnd
import networkx as nx
import plotly.graph_objects as go

from model.classes.layer import Layer
from model.classes.neuron import Neuron


class Network:
    def __init__(self, id=0):
        self.id = id
        self.layers = []

    def add_layer(self, num_of_neurons, propagation_function = 'weighted_sum', activation_function = 'identity', fixed_weight = None):
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

    def plot_network(self, show_nodes = True, show_edges = True):
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

    def get_error_dynamic(self, error_list, dynamic_range = 10):
        if len(error_list) < dynamic_range: return 0
        dynamic = 0
        for i in range(len(error_list) - dynamic_range, len(error_list)):
            dynamic += (error_list[i] - error_list[i - 1]) / abs(error_list[i - 1])
        dynamic /= dynamic_range
        return dynamic

    def create_validation_list(self,
            val_df_x,
            val_df_y):
            """Creates a list to validate the predictions of a model using a validation dataset as input."""
        
            val_data_x = val_df_x.values.tolist()
            val_data_y = val_df_y.values.tolist()
            result = []

            for i,v in enumerate(val_data_x):
                result.append([self.predict(v), val_data_y[i]])
                
            return result

class Feed_Forward(Network):
    def train(self,
        train_df_x,
        train_df_y,
        mode = 'online',
        epochs = 100,
        max_error = 0,
        adaptive_learning_rate = False,
        min_learning_rate = 0,
        default_learning_rate = 0.5,
        momentum_factor = 0,
        weight_decay_factor = 0,
        shuffle = False,
        debug = False):
        """Trains the model using normalized training data."""

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

            # add average Error to list of average Errors
            Err.append(Err_e_avg)
            if debug: print(f'epoch = {epoch} | average error = {Err_e_avg} | learning rate = {learning_rate}')

            # adapt learning rate based on error
            if adaptive_learning_rate:
                if self.get_error_dynamic(Err, 10) > 0:
                    learning_rate /= 2

            if Err[-1] < max_error:
                print(f'Training finished. Max error rate of < {max_error} reached on epoch {epoch}.')
                break

            if learning_rate < min_learning_rate:
                print(f'Training aborted. Learning rate reached < {min_learning_rate} on epoch {epoch}.')
                break
        
            if epoch == epochs:
                print(f'Training finished. Number of epochs reached.')

        return Err