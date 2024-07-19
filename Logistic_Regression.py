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

"""
Exercise 4: Initializing Parameters of the Neural Network
This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0
Argument: dim is the size of the w vector we want (number of parameters)
Returns: 
    w: initialized vector of shape (dim, 1)
    b: initialized scalar (corresponds to the bias) of type float
"""
print("Exercise 4: Initializing Parameters")
print("==========")
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
assert type(b) == float
print ("w = " + str(w))
print ("b = " + str(b))

initialize_with_zeros_test_1(initialize_with_zeros)
initialize_with_zeros_test_2(initialize_with_zeros)
print("========================================")

"""
Exercise 5: Forward and Backward Propagation
Now, we implement a function propagate() that computes the cost function and its gradient

- Forward propagation:
    - we get X
    - we compute A = 𝞼(wT X + b)
    - we calculate the cost function: J = - (1/m) 𝛴(y log(a) + (1 - y) log(1 - a))
    
- Arguments:
    - w: weights, a numpy array of size (num_px * num_px * 3, 1)
    - b: bias, a scalar
    - X: data of size (num_px * num_px * 3, number of examples)
    - Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

- Returns: 
    - grads: dictionary containing gradient of the weight and bias vectors
        (dw -- gradient of the loss with respect to w, thus same shape as w)
        (db -- gradient of the loss with respect to b, thus same shape as b)
    - cost: negative logarithm-likelihood of the cost for logistic regression

"""
print("Exercise 5: Propagate")
print("==========")
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = 1 / (1 + np.exp(-(np.dot(w.T, X) + b)))
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost

w =  np.array([[1.], [2]])
b = 1.5
X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])
grads, cost = propagate(w, b, X, Y)

assert type(grads["dw"]) == np.ndarray
assert grads["dw"].shape == (2, 1)
assert type(grads["db"]) == np.float64

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
propagate_test(propagate)
print("========================================")

"""
Exercise 6: Optimization
Now, after we initialized the parameters ad computed the cost function and its gradient, we want to update the parameters using gradient descent. 

The goal of this function is to learn w and b minimizing the cost function J using a gradient descent algorithm
For a parameter 𝞱, the update rule is: 𝞱 = 𝞱 - 𝞪d𝞱, where 𝞪 is the learning rate 

- Arguments:
    - w: weights, a numpy array of size (num_px * num_px * 3, 1)
    - b: bias, a scalar
    - X: data of size (num_px * num_px * 3, number of examples)
    - Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    - num_iterations: number of iterations of the optimization loop
    - learning_rate: learning rate of the gradient descent update rule
    - print_cost: True to print the loss every 100 steps

- Returns: 
    - params: dictionary containing the weights w and bias b
    - grads: dictionary containing the gradients of the weights and bias with respect to the cost function
    - costs: list of all the costs computed during the optimization, this will be used to plot the learning curve.
"""
print("Exercise 6: Optimize")
print("==========")
def optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # record the cost
        if i % 100 == 0:
            costs.append(cost)

            # print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("Costs = " + str(costs))
optimize_test(optimize)
print("========================================")
