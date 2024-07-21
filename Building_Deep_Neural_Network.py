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
# from testCases import *
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
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
"""
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1, 'b1':b1, 'W2': W2, 'b2': b2}

    return parameters

