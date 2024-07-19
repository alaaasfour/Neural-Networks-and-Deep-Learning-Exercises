# Logistic Regression with a Neural Network Mindset

# Packages
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import spicy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *


# Loading the data (cat/non-Cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 24
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


"""
Exercise 1:
We will find the values for:
- m_train: the number of training examples
- m_test: the number of test examples
- num_px: the number of pixels in the image (= height = width of the training image)
"""
m_train = len(train_set_x_orig)
m_test = len(test_set_x_orig)
num_px = train_set_x_orig.shape[2]
print("Exercise 1")
print("==========")
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print("========================================")

"""
Exercise 2:
Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape
(num_px * num_px * 3, 1)
"""

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print("Exercise 2")
print("==========")
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print("========================================")

"""
To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel 
value is actually a vector of three numbers ranging from 0 to 255.
"""
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


"""
Building the parts of the algorithm:
The main steps for building a Neural Network are:
    1. Defining the model structure (number of input features)
    2. Initialize the model parameters
    3. Loop:
        - Calculating the current loss (forward propagation)
        - Calculating the gradient (backward propagation)
        - Update parameters of the model (gradient descent)
        
We will build each step separately and combine them into a single function called model()
"""
"""
Exercise 3: Sigmoid Function
The sigmoid function has the formula: sigmoid(z) = 1 / (1 + e^{-z}) 
"""
print("Exercise 3: Sigmoid Function")
print("==========")
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
x = np.array([0.5, 0, 2.0])
output = sigmoid(x)
print(output)
sigmoid_test(sigmoid)
print("========================================")