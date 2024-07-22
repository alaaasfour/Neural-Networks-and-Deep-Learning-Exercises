# Deep Neural Network for Image Classification: ApplicationDeep
"""
For this lab, we will use the information of the 'Building Deep Neural Network.py' to build  cat/not-a-cat classifier.
We'll see an improvement in accuracy over the logistic regression implementation.
"""
import time
import h5py
import spicy
import numpy as np
from PIL import Image
from spicy import ndimage
from public_tests import *
from helper_mthods import *
import matplotlib.pyplot as plt

np.random.seed(1)

"""
Exercise 1: Load and Process the Dataset
We'll be using the same "Cat vs non-Cat" dataset as in "Logistic Regression as a Neural Network".
The model we built back then had 70% test accuracy on classifying cat vs non-cat images. 
This new model will perform even better!

"""
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
plt.show()
print("Exercise 1: Load and Process the Dataset")
print("==========")
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Explore the dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# We need to reshape and standardize the images before feeding them to the network.
# Note:  12,288  equals  64×64×3 , which is the size of one reshaped image vector.
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
print("========================================")



