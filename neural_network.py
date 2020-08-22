'''
Below is the Implementation of a feed forward neural network with Mean Squared Error as Loss function.
It contains both backPropagation and forwardPropagation implemented with mini-batch from scratch
All weights and biases and derivatives are saved in JSONs epoch wise.
'''

import numpy as np
import argparse
import json
from copy import deepcopy
import os


class NeuralNet:
    def __init__(self,batch_size, input_shape, epochs, layer_dim, learning_rate):
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.input_shape = [int(i) for i in input_shape.split(",")]
        self.layer_dim = [int(i) for i in layer_dim.split(",")]
        self.layer_dim.insert(0, self.input_shape[1])
        self.num_layer = len(self.layer_dim)
        self.cost = []
        self.parameters = {}

    @staticmethod
    def get_batches(input_data, batch_size):
        output_batch = []
        for i in range(0, len(input_data), batch_size):
            output_batch.append(input_data[i: i + batch_size])
        return output_batch

    @staticmethod
    def sigmoid(x):
        '''
        Description: Method to calculate sigmoid
        :param x: Input Matrix
        :return:
        '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        '''
        Description: Method to calculate derivative of sigmoid
        :param x: Input Matrix
        :return:
        '''
        sig_value = 1 / (1 + np.exp(-x))
        return sig_value * (1 - sig_value)

    @staticmethod
    def save_result_to_json(data, file_name, epoch):
        '''
        Description: Method to save the result in JSONs
        :param data: dictionary to be saved
        :param file_name: name of the file
        :param epoch: epoch count
        :return:
        '''

        if not os.path.isdir("output"):
            os.makedirs("output")

        for key, val in data.items():
            data[key] = data[key].tolist()

        try:
            with open(os.path.join("output", file_name+"_" + str(epoch) + ".json"), "w") as fd:
                json.dump(data, fd)
        except IOError as ioe:
            print("Unable to write {} for {}".format(file_name+".json", str(epoch)))

    def param_initialization(self):
        '''
        Description: Method to initialize parameters (weights and biases)
        :return:
        '''
        for i in range(1, self.num_layer):
            self.parameters["W" + str(i)] = np.random.randn(self.layer_dim[i-1], self.layer_dim[i]) / np.sqrt(self.layer_dim[i - 1])
            self.parameters["b" + str(i)] = np.zeros((1, self.layer_dim[i]))

    def forwardPropagation(self, X):
        '''
        Description: Method for forward Propagation in network
        :param X: Input Matrix
        :return: activations and IntermediateValues
        '''
        intermediate_values = {}
        activations = X

        for l in range(self.num_layer-1):
            Z = activations.dot(self.parameters["W" + str(l + 1)]) + self.parameters["b" + str(l + 1)]
            activations = NeuralNet.sigmoid(Z)
            intermediate_values["A" + str(l + 1)] = activations
            intermediate_values["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            intermediate_values["Z" + str(l + 1)] = Z

        return activations, intermediate_values

    def backPropagation(self, X, Y, intermediate_values):
        '''
        Description: Method for back propagation
        :param X: Input data matrix
        :param Y: Input target variable
        :param intermediate_values: IntermediateValues came from forward propagation
        :return: return derivatives
        '''

        derivatives = {}

        intermediate_values["A0"] = X
        A = intermediate_values["A" + str(self.num_layer-1)]

        dA = (Y - A) * NeuralNet.sigmoid_derivative(A)
        dZ = dA * self.sigmoid_derivative(intermediate_values["Z" + str(self.num_layer-1)])

        dW = intermediate_values["A" + str(self.num_layer - 2)].T.dot(dZ)/len(Y)

        db = np.sum(dZ, axis=0, keepdims=True)/len(Y)
        dAPrev = dZ.dot(intermediate_values["W" + str(self.num_layer-1)].T)

        derivatives["dW" + str(self.num_layer-1)] = dW
        derivatives["db" + str(self.num_layer-1)] = db

        for i in range(self.num_layer -2, 0, -1):
            dZ = dAPrev * NeuralNet.sigmoid_derivative(intermediate_values["Z" + str(i)])
            dW = 1. / len(Y) * intermediate_values["A" + str(i - 1)].T.dot(dZ)
            db = 1. / len(Y) * np.sum(dZ, axis=0, keepdims=True)

            if i > 1:
                dAPrev = dZ.dot(intermediate_values["W" + str(i)].T)

            derivatives["dW" + str(i)] = dW
            derivatives["db" + str(i)] = db

        return derivatives

    def fit(self):
        #Initialize data with random values
        X_input_data = np.random.rand(self.input_shape[0], self.input_shape[1])
        Y_input_data = np.random.rand(self.input_shape[0], 1)

        #create batches
        X_batches = NeuralNet.get_batches(X_input_data, self.batch_size)
        Y_batches = NeuralNet.get_batches(Y_input_data, self.batch_size)

        #initialize parameters(weight_and_biases)
        self.param_initialization()

        #run the model
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            epoch_cost = []
            derivatives = {}
            for i in range(len(X_batches)):
                A, intermediate_values = self.forwardPropagation(X_batches[i])
                print("ForwardPropagation completed.")
                cost = (sum((A - Y_batches[i])**2))/len(Y_batches[i])
                epoch_cost.append(cost)
                derivatives = self.backPropagation(X_batches[i], Y_batches[i], intermediate_values)
                print("Back propagation completed.")
                for l in range(1, self.num_layer):
                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate * derivatives["dW" + str(l)]
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate * derivatives["db" + str(l)]
            avg_cost = sum(epoch_cost)/len(X_batches)
            self.cost.append(avg_cost)

            #save the weights and biases
            weight_and_biases = deepcopy(self.parameters)
            NeuralNet.save_result_to_json(weight_and_biases, "weight_and_biases", epoch)

            #save derivatives
            derivatives_ = deepcopy(derivatives)
            NeuralNet.save_result_to_json(derivatives_, "derivatives", epoch)
            print("Epoch {} completed".format(str(epoch)))
            print("\n")


def main():
    parser = argparse.ArgumentParser(description="Neural Network")
    parser.add_argument("-b", "--batch_size", help="specify batch size", default="64", required=False)
    parser.add_argument("-s", "--input_shape", help="specify input shape  separated by ',' ", required=True)
    parser.add_argument("-n", "--epochs", help="number of epochs", required=False, default=10)
    parser.add_argument("-l", "--layer_dim", help="layer dimensions separated by ',' ", required=True)
    parser.add_argument("-r", "--learning_rate", help="learning rate for the network", required=False, default=0.001)

    args = parser.parse_args()

    batch_size = args.batch_size
    input_shape = args.input_shape
    epochs = args.epochs
    layer_dim = args.layer_dim
    learning_rate = args.learning_rate

    print("batch_size: {}".format(batch_size))
    print("epochs: {}".format(epochs))
    print("learning_rate: {}".format(learning_rate))
    print("layer_dim: {}".format(layer_dim))
    print("input_shape: {}".format(input_shape))

    network = NeuralNet(batch_size, input_shape, epochs, layer_dim, learning_rate)
    network.fit()


if __name__ == '__main__':
    main()