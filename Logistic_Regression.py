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
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))