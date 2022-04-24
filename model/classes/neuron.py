from model import helpers as hlp

class Neuron:
    # constructor
    def __init__(self, id):
        '''Creates a neuron object.
        
        Parameters
        ----------
            id (int): ID of the neuron.
        '''
        self.id = id
        self.net = 0.0
        self.activation = 0.0
        self.output = 0.0 if self.id[1] != 0 else 1.0 # bias neurons have id 0 and get an output of 1
        self.delta = 0.0
    

    # main functions
    def do_update(self, prev_layer_neurons, current_layer):
        '''Updates a neurons output value by propagating and activating.

        Parameters
        ----------
            prev_layer_neurons (list): list of neurons of the previous layer.
            current_layer (model.Layer): Layer, that the neuron is part of.
        
        Returns:
        ----------
            successful (bool): `False`, if the updating was not successful, else `True`.
        '''
        if not self.do_propagate(prev_layer_neurons, current_layer): return False
        if not self.do_activate(current_layer.activation_function): return False
        if not self.do_output(): return False
        return True

    def do_propagate(self, prev_layer_neurons, current_layer):
        '''Computes a neurons net input by using a propagation function.

        Parameters
        ----------
            prev_layer (model.Layer): Previous layer of the network model
            current_layer (model.Layer): Layer, that the neuron is part of
        
        Returns:
        ----------
            successful (bool): `False`, if the updating was not successful, else `True`.
        '''
        net_input = 0.0

        if current_layer.propagation_function == 'weighted_sum':
            for n in prev_layer_neurons:
                net_input += n.output * current_layer.weights[n.id[1]][self.id[1] - 1]

            self.net = net_input
            return True
        else:
            return False

    def do_activate(self, activation):
        '''Computes the activation value using the net input and the layers activation function.

        Parameters
        ----------
            activation (string): Name of the activation function that is to be used. Options are `'identity'`, `'relu'`, `'sigmoid'`, `'tanh'`
        
        Returns:
        ----------
            successful (bool): `False`, if the updating was not successful, else `True`.
        '''
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
        '''Computes the neuron's output using it's activation value. Currently the identity function is used.
        
        Returns:
        ----------
            successful (bool): `False`, if the updating was not successful, else `True`.
        '''
        if (self.id[1] == 0):
            return False

        self.output = self.activation
        return True

    def do_activate_der(self, activation_function):
        '''Computes the the value for the derivative of a neurons activation using the net input.
        
        Parameters:
        ----------
            activation (string): Name of the activation function that is to be used. Options are `'identity'`, `'relu'`, `'sigmoid'`, `'tanh'` 

        Returns:
        ----------
            result (float): Result of the calculation
        '''
        match activation_function:
            case 'identity':
                return hlp.activate_identity_der(self.net)
            case 'relu':
                return hlp.activate_relu_der(self.net)
            case 'sigmoid':
                return hlp.activate_sigmoid_der(self.net) 
            case 'tanh':
                return hlp.activate_tanh_der(self.net)