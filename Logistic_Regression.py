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
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

