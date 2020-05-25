from random import random


# in our case relu = linear
def relu(x):
    if x <= 0:
        return 0
    else:
        return x


# only for x > 0
def derivative_relu(x):
    return 1


def adder(inputs, weights):
    if len(inputs) != len(weights):
        raise IndexError
    tmp = 0
    for x in range(len(inputs)):
        tmp += inputs[x]*weights[x]
    return tmp


def mean_square_error(output_data, expected_data):
    error = 0.0
    elements_in_output = 0
    for i in range(len(output_data)):
        for j in range(len(output_data[i])):
            elements_in_output = len(output_data[i])
            error += (output_data[i][j]-expected_data[i][j]) * (output_data[i][j]-expected_data[i][j])

    return error/(len(output_data)*elements_in_output)


class Neuron:
    def __init__(self, act_fun):
        self.inputs = []
        self.weights = []
        self.activation_function = act_fun

    def work(self):
        return self.activation_function(adder(self.inputs, self.weights))

    def work_for_input_layer(self):
        return self.inputs

    def init_weights(self):
        for x in range(len(self.inputs)):
            self.weights.append(random())


class Layer:
    def __init__(self, amount_of_neurons, act_fun, input_layer_flag, previous_layer):
        self.input_layer_flag = input_layer_flag
        self.neurons = []
        for x in range(amount_of_neurons):
            self.neurons.append(Neuron(act_fun))
        if not input_layer_flag:
            self.previous_layer = previous_layer

    def send_output(self):
        outputs = []
        for neuron in self.neurons:
            if not self.input_layer_flag:
                outputs.append(neuron.work())
            else:
                outputs.append(neuron.work_for_input_layer())
        return outputs

    def load_input_for_first_layer(self, data):
        for x in range(len(data)):
            self.neurons[x].inputs = data[x]
            if len(self.neurons[x].weights) < 1:
                self.neurons[x].weights.append(1)

    def load_data_from_previous_layer(self):
        for neuron in self.neurons:
            neuron.inputs = self.previous_layer.send_output()
            if len(neuron.weights) == 0:
                neuron.init_weights()


class NeuralNetwork:
    def __init__(self, amount_of_layers, neurons_in_layer):
        self.layers = []
        for i in range(amount_of_layers):
            if i == 0:
                self.layers.append(Layer(neurons_in_layer[i],relu,True,None))
            else:
                self.layers.append(Layer(neurons_in_layer[i], relu, False, self.layers[-1]))

    def get_output(self):
        return self.layers[-1].send_output()

    def train(self, inputs):
        self.layers[0].load_input_for_first_layer(inputs)
        first = True
        for layer in self.layers:
            if first:
                first = False
                continue
            layer.load_data_from_previous_layer()

res =[]
expected = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
siec = NeuralNetwork(3, [4, 4, 4])
for epochs in range(5):
    siec.train([1, 2, 3, 4])
    res.append(siec.get_output())
print(mean_square_error(res,expected))






