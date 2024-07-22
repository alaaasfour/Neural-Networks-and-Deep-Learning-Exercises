# Logistic Regression with a Neural Network 🪄

## Description 📖
### This project implements a logistic regression classifier to recognize cats in images. The main tasks include:

### 1. Building the general architecture of a learning algorithm:
* Initializing parameters
* Calculating the cost function and its gradient
* Using an optimization algorithm (gradient descent)

### 2. Combining all three functions into a main model function, in the correct order.
<br>To run: `Logistic_Regression.py`

## Getting Started 🚀

## Prerequisites 🐍
Make sure you have Python 3.10+ installed on your machine. In addition to the following packages:
* `numpy`
* `h5py`
* `PIL`
* `scipy`
* `matplotlib`

## Dataset 🗳️
The dataset (`data.h5`) contains:
* A training set of images labeled as cat (y=1) or non-cat (y=0)
* A test set of images labeled as cat or non-cat
* Each image is of shape (num_px, num_px, 3) where 3 represents the RGB channels.

## File Structure 🗂️
* `lr_utils.py`: Contains the load_dataset function to load the data.
* `public_tests.py`: Contains test functions for validating each step of the implementation.


## Features ✨
1. Sigmoid Function
2. Initialize Parameters
3. Forward and Backward Propagation
4. Optimization
5. Prediction
6. Model


## Screenshots 🖼️
![Screenshot 2024-07-19 at 8 01 20 PM](https://github.com/user-attachments/assets/7aba6cdb-b5f3-4949-bfb3-c2cd059f08c2)
![Screenshot 2024-07-19 at 8 01 36 PM](https://github.com/user-attachments/assets/aad25379-582a-41d1-b17c-0579d3241c53)
![Screenshot 2024-07-19 at 8 01 49 PM](https://github.com/user-attachments/assets/e2b198f2-0d8d-4f03-909f-1b36dc24e49a)
![Screenshot 2024-07-19 at 8 02 02 PM](https://github.com/user-attachments/assets/1c2588b4-589e-4e3e-b9a7-a1a3b619002d)

# Building Deep Neural Network 🪄🕸️

## Description 📖
### This project demonstrates the step-by-step implementation of a deep neural network with multiple layers using Python. The primary goal is to understand and build the foundational functions required for constructing and training a deep neural network.

## Prerequisites 🐍
Make sure you have Python 3.10+ installed on your machine. In addition to the following packages:
* `numpy`
* `h5py`
* `PIL`
* `scipy`
* `matplotlib`

You can install these packages using pip: `pip install numpy h5py matplotlib pillow scipy`

## Features & Exercises ✨☄️
1. Initializing Parameters: The function `initialize_parameters` initializes the parameters for a 2-layer neural network.
2. Initializing Parameters Deep: The function `initialize_parameters_deep` initializes parameters for an L-layer neural network.
3. Forward Propagation Module: The function `linear_forward` computes the linear part of forward propagation.
4. Linear Activation Forward: The function `linear_activation_forward` computes the forward propagation with activation functions (ReLU and Sigmoid).
5. L-Layer Model: The function `L_model_forward` implements the forward propagation for the entire L-layer neural network.
6. Cost Function: The function `compute_cost` calculates the cost using cross-entropy.
7. Backward Propagation Module: The function `linear_backward` computes the linear part of the backward propagation.
8. Linear-Activation Backward: The function `linear_activation_backward` implements the backward propagation with activation functions (ReLU and Sigmoid).
9. L-Model Backward: The function `L_model_backward` implements the backward propagation for the entire L-layer neural network.
10. Update Parameters: The function `update_parameters` updates the network parameters using gradient descent.