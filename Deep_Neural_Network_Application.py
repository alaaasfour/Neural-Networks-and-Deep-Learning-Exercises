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

"""
Exercise 2: Two-layer Neural Network
It's time to build a deep neural network to distinguish cat images from non-cat images!
we're going to build two different models:
    - A 2-layer neural network
    - An L-layer deep neural network

General Methodology
As usual, we'll follow the Deep Learning methodology to build the model:
    1. Initialize parameters / Define hyperparameters
    2. Loop for num_iterations: a. Forward propagation b. Compute cost function c. Backward propagation d. Update parameters (using parameters, and grads from backprop)
    3. Use trained parameters to predict labels
    
Arguments:
    X: input data, of shape (n_x, number of examples)
    Y: true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims: dimensions of the layers (n_x, n_h, n_y)
    num_iterations: number of iterations of the optimization loop
    learning_rate: learning rate of the gradient descent update rule
    print_cost: If set to True, this will print the cost every 100 iterations 
    
Returns:
    parameters: a dictionary containing W1, W2, b1, and b2
"""
### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost = False):
    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

print("Exercise 2: Load and Process the Dataset")
print("==========")
parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2, print_cost = False)
print("Cost after first iteration: " + str(costs[0]))

two_layer_model_test(two_layer_model)
print("========================================")

"""
Exercise 3: Train the model
"""
# Now, we will train the parameters
parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost = True)
plot_costs(costs, learning_rate)

# After training the model, we can use the trained parameters to classify images from the dataset.
# Let's know the accuracy of the trained model:
predictions_train = predict(train_x, train_y, parameters)
# Let's know the accuracy of the test model:
predictions_test = predict(test_x, test_y, parameters)

# We can notice that the 2-layer neural network has better performance (72%) than the logistic regression implementation.

"""
Exercise 4: L-layer Neural Network
We're going to build the L-layer deep neural network


Arguments:
    X: input data, of shape (n_x, number of examples)
    Y: true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
Returns:
    parameters: parameters learnt by the model. They can then be used to predict.
"""
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost = False):
    np.random.seed(1)
    costs = []  # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

print("Cost after first iteration: " + str(costs[0]))

L_layer_model_test(L_layer_model)

"""
Exercise 5: Train the model for L-Layer Neural Network
"""
# Now, we will train the parameters
parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

# After training the model, we can use the trained parameters to classify images from the dataset.
# Let's know the accuracy of the trained model:
pred_train = predict(train_x, train_y, parameters)
# Let's know the accuracy of the test model:
pred_test = predict(test_x, test_y, parameters)

# We can notice that it seems that the 4-layer neural network has better performance (80%) than the 2-layer neural network (72%) on the same test set.