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
print("Exercise 1: Initializing Parameters")
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

print("Exercise 2: Initializing Parameters Deep")
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

print("Exercise 3: Forward Propagation Module")
print("==========")
t_A, t_W, t_b = linear_forward_test_case()
t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
print("Z = " + str(t_Z))

linear_forward_test(linear_forward)
print("========================================")

"""
Exercise 4: Linear Activation Forward
To implement the linear activation forward, we will use two activation functions:
1. Sigmoid function: which returns two items; the activation value 'a' and a 'cache'
2. ReLU function: which returns two items; the activation value 'A' and a 'cache' that contains 'Z'

Arguments:
    A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
    W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b: bias vector, numpy array of shape (size of the current layer, 1)
    activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

Returns:
    A: the output of the activation function, also called the post-activation value 
    cache: a python tuple containing "linear_cache" and "activation_cache"; stored for computing the backward pass efficiently
"""

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

print("Exercise 4: Linear Activation Forward")
print("==========")
t_A_prev, t_W, t_b = linear_activation_forward_test_case()

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "sigmoid")
print("With sigmoid: A = " + str(t_A))

t_A, t_linear_activation_cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "relu")
print("With ReLU: A = " + str(t_A))

linear_activation_forward_test(linear_activation_forward)
print("========================================")

"""
Exercise 5: L-Layer Model
For even more convenience when implementing the  L-layer Neural Net, we will need a function that replicates 
the previous one (linear_activation_forward with RELU)  L−1  times, then follows that with one linear_activation_forward with SIGMOID.

Arguments:
    X: data, numpy array of shape (input size, number of examples)
    parameters: output of initialize_parameters_deep()

Returns:
    AL: activation value from the output (last) layer
    caches: list of caches containing every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
"""

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network

    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

print("Exercise 5: L-Layer Model")
print("==========")
t_X, t_parameters = L_model_forward_test_case_2hidden()
t_AL, t_caches = L_model_forward(t_X, t_parameters)

print("AL = " + str(t_AL))

L_model_forward_test(L_model_forward)
print("========================================")


"""
Exercise 6: Cost Function
We need to implement the cost, in order to check whether the model is actually learning

Arguments:
    AL: probability vector corresponding to your label predictions, shape (1, number of examples)
    Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

Returns:
    cost: cross-entropy cost
"""

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.shape(cost)

    return cost

print("Exercise 6: Cost Function")
print("==========")
t_Y, t_AL = compute_cost_test_case()
t_cost = compute_cost(t_AL, t_Y)

print("Cost: " + str(t_cost))

compute_cost_test(compute_cost)
print("========================================")

"""
Exercise 7: Backward Propagation Module
Similarly to the forward propagation, we need to implement the backpropagation.
Backpropagation is used to calculate the gradient of the loss function with respect to the parameters.
Arguments:
    dZ: Gradient of the cost with respect to the linear output (of current layer l)
    cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

Returns:
    dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: Gradient of the cost with respect to W (current layer l), same shape as W
    db: Gradient of the cost with respect to b (current layer l), same shape as b
"""

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

print("Exercise 7: Backward Propagation Module")
print("==========")
t_dZ, t_linear_cache = linear_backward_test_case()
t_dA_prev, t_dW, t_db = linear_backward(t_dZ, t_linear_cache)

print("dA_prev: " + str(t_dA_prev))
print("dW: " + str(t_dW))
print("db: " + str(t_db))

linear_backward_test(linear_backward)
print("========================================")
