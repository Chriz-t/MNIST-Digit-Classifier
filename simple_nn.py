"""
Neural Network for MNIST Digit Recognition
==========================================

PRESENTATION
------------
This Python program implements a simple feedforward neural network 
from scratch using only:
    - NumPy (matrix operations and math)
    - Pandas (data loading)
    - Matplotlib (visualization of digits)

The network is trained on the MNIST dataset of handwritten digits 
(28x28 grayscale images). All computations (forward propagation, 
backpropagation, weight updates, and loss calculation) are coded 
manually without using machine learning frameworks.

------------------------------------------
STRUCTURE OF THE NEURAL NETWORK
------------------------------------------
This is a forward propagating neural network with 3 layers:
    - Input layer: 784 nodes (28 x 28 flattened pixels)
    - Hidden layer: 10 nodes (with ReLU activation)
    - Output layer: 10 nodes (with Softmax activation)

Key details:
    - Activation functions:
        * ReLU for the hidden layer
        * Softmax for the output layer
    - Loss function: Cross-Entropy Loss
    - Optimizer: Batch Gradient Descent
    - Parameters are initialized randomly and updated iteratively

------------------------------------------
USAGE GUIDE
------------------------------------------
1. Training:
   The network is trained using the function `gradient_descent()`.
   During training, accuracy and loss are printed every 50 iterations.

2. Prediction:
   Use `make_prediction()` to classify one or multiple samples.

3. Visualization:
   The function `test_prediction()` displays the digit image along 
   with the predicted label.

4. Evaluation:
   After training, predictions are made on a test split and overall 
   accuracy is displayed.

------------------------------------------
NOTATIONS
------------------------------------------
x       : Input data (features)
y       : True labels
z1, z2  : Linear outputs of hidden and output layers
a1, a2  : Activations of hidden and output layers
w1, w2  : Weight matrices of layer 1 and 2
b1, b2  : Bias vectors of layer 1 and 2
dz1, dz2: Error terms during backpropagation
dw1, dw2: Gradients of weights
db1, db2: Gradients of biases
alpha   : Learning rate
m       : Number of training samples

------------------------------------------
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# -----------------------
# Data Preprocessing
# -----------------------

# Load dataset from CSV (Kaggle MNIST format).
# The first column is the label, the rest are pixel values (28x28=784).
data = pd.read_csv("Train data path")
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)  # Shuffle to mix digits randomly

# Split into test (first 1000) and train (rest)
data_test = data[0:1000].T
y_test = data_test[0]          # labels
x_test = data_test[1:n] / 255  # normalize to [0,1]

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255


# -----------------------
# Helper Functions
# -----------------------

def ReLU(z):
    # ReLU sets all negative values to 0.
    # This introduces non-linearity and avoids vanishing gradients.
    return np.maximum(z, 0)


def softmax(z):
    # Softmax converts raw scores into probabilities for each class.
    # Subtracting max(z) improves numerical stability.
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def init_params():
    # Initialize weights and biases randomly.
    # Small random numbers help break symmetry between neurons.
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def forward_prop(w1, b1, w2, b2, x):
    # Forward pass: compute activations layer by layer.
    # z1 = input to hidden, a1 = activation of hidden
    # z2 = hidden to output, a2 = final softmax probabilities
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def one_hot(y):
    # One-hot encoding converts labels like "3" into [0,0,0,1,0,...].
    # This is required for computing cross-entropy loss.
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T


def derv_ReLU(z):
    # Derivative of ReLU: 1 for positive inputs, 0 for negatives.
    # Used during backpropagation.
    return z > 0


def back_prop(z1, a1, z2, a2, w2, x, y):
    # Backpropagation computes gradients of loss w.r.t. weights and biases.
    # Errors flow backward: output layer → hidden layer → input layer.
    m = y.size
    one_y = one_hot(y)

    # Error at output layer (softmax - one-hot labels)
    dz2 = a2 - one_y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

    # Error at hidden layer (propagated through w2 and ReLU derivative)
    dz1 = w2.T.dot(dz2) * derv_ReLU(z1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    # Gradient descent update rule:
    # new_param = old_param - learning_rate * gradient
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    # The predicted digit is the index of the highest probability.
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    # Accuracy = correctly predicted samples / total samples
    return np.sum(predictions == y) / y.size


def compute_loss(a2, y):
    # Cross-entropy loss measures how far predictions are from true labels.
    # Adding small epsilon avoids log(0) errors.
    # Not used to train the model but to display the loss.
    m = y.size
    oh_Y = one_hot(y)
    return -1/m * np.sum(oh_Y * np.log(a2 + 1e-8))


def gradient_descent(x, y, iterations, alpha):
    # Train the neural network using gradient descent.
    # In each iteration: forward pass → backpropagation → update parameters.
    w1, b1, w2, b2 = init_params()

    for i in range(iterations + 1):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        # Print progress every 50 iterations
        if i % 50 == 0:
            preds = get_predictions(a2)
            acc = get_accuracy(preds, y)
            loss = compute_loss(a2, y)
            print(f"Iteration: {i}, Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    return w1, b1, w2, b2


def make_prediction(x, w1, b1, w2, b2):
    # Predict digit(s) for given input.
    # Runs forward propagation and takes argmax of output.
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, x)
    return get_predictions(a2)


def test_prediction(index, w1, b1, w2, b2):
    # Test prediction for a single sample and visualize it.
    # Shows predicted label, true label, and the image.
    current_image = x_train[:, index, None]
    prediction = make_prediction(current_image, w1, b1, w2, b2)
    label = y_train[index]
    print(f"Prediction: {prediction}, Label: {label}")

    current_image = current_image.reshape(28, 28) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# -----------------------
# Training
# -----------------------
w1, b1, w2, b2 = gradient_descent(x_train, y_train, iterations=1000, alpha=0.1)

# -----------------------
# Testing
# -----------------------
test_prediction(5, w1, b1, w2, b2)
test_predictions = make_prediction(x_test, w1, b1, w2, b2)
print("Test Accuracy:", get_accuracy(test_predictions, y_test))

