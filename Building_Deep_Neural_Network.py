# Building Deep Neural Network Step by Step
"""
In this lab, we will implement all the required functions to build a deep neural network with multiple layers.
"""

# Packages
"""
h5py is a common package to interact with a dataset that is stored on an H5 file.
PIL and scipy are used here to test the model with different pictures from the local machine.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from public_tests import *
import copy
np.random.seed(1)

"""
Exercise 1: Initializing Parameters
We will create and initialize the parameters of the 2-layer neural network.
The model structure will look like this: LINEAR -> RELU -> LINEAR -> SIGMOID
Argument:
    n_x: size of the input layer
    n_h: size of the hidden layer
    n_y: size of the output layer
    
Returns:
    parameters: python dictionary containing your parameters:
                    W1: weight matrix of shape (n_h, n_x)
                    b1: bias vector of shape (n_h, 1)
                    W2: weight matrix of shape (n_y, n_h)
                    b2: bias vector of shape (n_y, 1)
"""
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1, 'b1':b1, 'W2': W2, 'b2': b2}

    return parameters
print("Exercise 1")
print("==========")
print("Test Case 1:\n")
parameters = initialize_parameters(3,2,1)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_test_1(initialize_parameters)

print("\033[90m\nTest Case 2:\n")
parameters = initialize_parameters(4,3,2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_test_2(initialize_parameters)
print("========================================")

"""
Exercise 2: Initializing Parameters Deep
We will implement the initializing for L-layer neural network.
The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. 
When completing the initialize_parameters_deep function, we should make sure that dimensions match between each layer.
The model structure will look like this: *[LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID*
this model has  L−1  layers using a ReLU activation function followed by an output layer with a sigmoid activation function.

Argument:
    layer_dims: python array (list) containing the dimensions of each layer in our network

Returns:
    parameters: python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                    Wl: weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl: bias vector of shape (layer_dims[l], 1)
"""

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))

        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))

    return parameters

print("Exercise 2")
print("==========")
print("Test Case 1:\n")
parameters = initialize_parameters_deep([5,4,3])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_deep_test_1(initialize_parameters_deep)

print("\033[90m\nTest Case 2:\n")
parameters = initialize_parameters_deep([4,3,2])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
initialize_parameters_deep_test_2(initialize_parameters_deep)
print("========================================")

"""
Exercise 3: Forward Propagation Module
After initializing the parameters, the forward propagation module can be done. 
The linear forward module computes the following equation: Z[l] = W[l]A[l-1]+b[l]

Argument:
    A: activations from previous layer (or input data): (size of previous layer, number of examples)
    W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b: bias vector, numpy array of shape (size of the current layer, 1)

Returns:
    Z: the input of the activation function, also called pre-activation parameter 
    cache: a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
"""
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

print("Exercise 3")
print("==========")
t_A, t_W, t_b = linear_forward_test_case()
t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
print("Z = " + str(t_Z))

linear_forward_test(linear_forward)
print("========================================")

