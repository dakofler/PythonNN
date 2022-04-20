from model import helpers as hlp


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
