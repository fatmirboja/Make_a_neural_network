import numpy as np

# Simple network with only one neuron, which has three input connections and one
# output connection
#  O \
#  O - O -
#  O /
#
class NeuralNetwork():

    def __init__(self):
        # radom seed
        np.random.seed(1)

        # input weights
        self.input_weights = 2 * np.random.random((3,1)) - 1

    # sigmoid activation function
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # computes the output
    def forward_pass(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.input_weights))

    # training function
    def train(self, training_set_inputs, training_set_outputs, iterations):

        # go through each iteration
        for _ in range(iterations):
            # Forward pass
            output = self.forward_pass(training_set_inputs)

            # Error
            error = training_set_outputs - output

            # Backpropagation
            gradient = error * self.__sigmoid_derivative(output)
            print('Gradient \n',gradient)
            adjustment = np.dot(training_set_inputs.T, gradient)
            print('Adjustment \n', adjustment)

            # Weights adjustment
            self.input_weights += adjustment
            print(self.input_weights)

# create the neural network
nn = NeuralNetwork()
# print initial weights
print('Initial weights')
print(nn.input_weights)

# The training set. We have 4 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = np.array([[0, 1, 1, 0]]).T

# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
nn.train(training_set_inputs, training_set_outputs, 3)

print('Trained weights')
print(nn.input_weights)

# Test the neural network with a new situation.
print("Considering new situation [1, 0, 0] -> ?: ")
print(nn.forward_pass(np.array([1, 0, 0])))
