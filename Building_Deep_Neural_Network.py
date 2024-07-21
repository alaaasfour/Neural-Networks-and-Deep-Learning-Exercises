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

