#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 2025
@author: sohampradhan
"""
# Import necessary libraries
import os                      # For file path operations
import numpy as np             # For numerical operations
import pandas as pd            # For data manipulation and reading CSV files
from matplotlib import pyplot as plt  # For visualization

# --- LOADING AND PREPARING THE DATA ---
# Expand the tilde character to get the full path to the user's Downloads directory
train_csv_path = os.path.expanduser("~/Downloads/train.csv")
# test_csv_path = os.path.expanduser("~/Downloads/test.csv")  # Commented out but could be used later

# Read the training data from CSV file and convert to numpy array
data = pd.read_csv(train_csv_path).values
# Get dimensions of the data: m = number of examples, n = number of features (including label)
m, n = data.shape
# Shuffle the data randomly to ensure even distribution of digits in training
np.random.shuffle(data)

# --- SPLIT DATA INTO DEVELOPMENT AND TRAINING SETS ---
# Take first 1000 examples as development set and transpose
# After transposition, each column represents one example
data_dev = data[:1000].T

# Extract labels (first row after transposition) and convert to integers
Y_dev = data_dev[0].astype(int)

# Extract features (remaining rows) and normalize by dividing by 255
# Normalization is critical for neural networks to work effectively
X_dev = data_dev[1:] / 255.0  # Pixel values now range from 0 to 1 instead of 0 to 255

# Use remaining examples for training set, with same preprocessing steps
data_train = data[1000:].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:] / 255.0  # Normalization is key for stable training

# Nueral network helper functions

def init_params():
    """
    Initialize neural network parameters with careful weight initialization
    
    Returns:
        W1: Weights for first layer (10×784)
        b1: Biases for first layer (10×1)
        W2: Weights for second layer (10×10)
        b2: Biases for second layer (10×1)
    """
    # The initialization for weights - scaled by square root of number of inputs
    # This helps prevent vanishing/exploding gradients
    W1 = np.random.randn(10, 784) * np.sqrt(1. / 784)  # First layer weights
    
    b1 = np.zeros((10, 1))  # Initialize biases to zero (better than random)
    
    W2 = np.random.randn(10, 10) * np.sqrt(1. / 10)    # Second layer weights
    
    b2 = np.zeros((10, 1))  # Second layer biases
    return W1, b1, W2, b2

def relu(Z):
    """
    Rectified Linear Unit activation function
    Returns Z if Z > 0, otherwise returns 0
    
    Args:
        Z: Input pre-activation matrix
        
    Returns:
        Activated values with non-linear transformation
    """
    return np.maximum(0, Z)  # Element-wise maximum between 0 and Z

def relu_deriv(Z):
    """
    Derivative of ReLU function for backpropagation
    Returns 1 if Z > 0, otherwise returns 0
    
    Args:
        Z: Input pre-activation matrix
        
    Returns:
        Derivative values (1 for positive inputs, 0 for negative)
    """
    return (Z > 0).astype(float)  # Convert boolean (True/False) to float (1.0/0.0)

def softmax(Z):
    """
    Softmax activation function for output layer
    Converts raw scores to probabilities that sum to 1
    
    Args:
        Z: Input pre-activation matrix
        
    Returns:
        Probability distribution for each example (column)
    """
    # Subtract max value for numerical stability (prevents overflow from exponentials)
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    
    # Divide by sum to get normalized probabilities (each column sums to 1)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(Y):
    """
    Convert label vector Y of shape (m,) into one-hot encoded matrix 
    of shape (num_classes, m).
    
    Args:
        Y: Vector of integer class labels
        
    Returns:
        One-hot encoded matrix where each column represents an example
    """
    classes = Y.max() + 1    # Number of classes (assuming 0-indexed)
    
    m = Y.size               # Number of examples
    
    one_hot_Y = np.zeros((classes, m))  # Initialize matrix of zeros
    
    # Set the correct positions to 1 (the row corresponding to the class label)
    one_hot_Y[Y, np.arange(m)] = 1
    
    return one_hot_Y

# Forwards and backwards propegation

def forward_prop(W1, b1, W2, b2, X):
    """
    Forward propagation through the neural network
    
    Args:
        W1, b1: First layer weights and biases
        W2, b2: Second layer weights and biases
        X: Input data (features)
        
    Returns:
        Z1: First layer pre-activations
        A1: First layer activations
        Z2: Second layer pre-activations
        A2: Second layer activations (final output probabilities)
    """
    # First layer computation: Z1 = W1·X + b1
    Z1 = W1.dot(X) + b1  # Matrix multiplication of weights and inputs plus bias
    A1 = relu(Z1)        # Apply ReLU activation function
    
    # Second layer computation: Z2 = W2·A1 + b2
    Z2 = W2.dot(A1) + b2  # Matrix multiplication of weights and first activations plus bias
    A2 = softmax(Z2)      # Apply softmax for probability distribution
    
    return Z1, A1, Z2, A2  # Return all activations and pre-activations for backprop

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    """
    Backward propagation to compute gradients
    
    Args:
        Z1: First layer pre-activations
        A1: First layer activations
        Z2: Second layer pre-activations
        A2: Second layer activations (output probabilities)
        W2: Second layer weights
        X: Input data
        Y: True labels
        
    Returns:
        dW1, db1: Gradients for first layer weights and biases
        dW2, db2: Gradients for second layer weights and biases
    """
    m = Y.size  # Number of examples
    
    # Convert labels to one-hot format for loss calculation
    Y_encoded = one_hot(Y)
    
    # Output layer error (derivative of softmax cross-entropy loss)
    # A2 - Y_encoded gives us the error at the output layer
    dZ2 = A2 - Y_encoded
    
    # Compute gradients for second layer
    dW2 = (1 / m) * dZ2.dot(A1.T)  # Scale gradient by 1/m and multiply by A1 transpose
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Average across all examples
    
    # Backpropagate error to first layer
    # W2.T.dot(dZ2) gives error contribution to A1
    # Multiply by derivative of ReLU to get error at Z1
    dZ1 = W2.T.dot(dZ2) * relu_deriv(Z1)
    
    # Compute gradients for first layer
    dW1 = (1 / m) * dZ1.dot(X.T)  # Scale by 1/m and multiply by X transpose
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Average across examples
    
    return dW1, db1, dW2, db2  # Return all gradients

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update parameters using gradient descent
    
    Args:
        W1, b1, W2, b2: Current parameters
        dW1, db1, dW2, db2: Computed gradients
        alpha: Learning rate
        
    Returns:
        Updated parameters after one step of gradient descent
    """
    # Update each parameter by subtracting the learning rate times the gradient
    W1 -= alpha * dW1  # Update first layer weights
    b1 -= alpha * db1  # Update first layer biases
    W2 -= alpha * dW2  # Update second layer weights
    b2 -= alpha * db2  # Update second layer biases
    
    return W1, b1, W2, b2  # Return updated parameters

# Training and evaluation functions

def get_predictions(A2):
    """
    Get class predictions from output probabilities
    
    Args:
        A2: Output probabilities from softmax
        
    Returns:
        Predicted digit for each example
    """
    
    # For each example, pick the class with highest probability
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    """
    Calculate accuracy of predictions
    
    Args:
        predictions: Predicted digits
        Y: True labels
        
    Returns:
        Accuracy as proportion of correct predictions
    """
    
    # Calculate mean of correct predictions (1 if prediction matches true label, 0 otherwise)
    return np.mean(predictions == Y)

def gradient_descent(X, Y, iterations, alpha):
    """
    Main training function implementing gradient descent
    
    Args:
        X: Training features
        Y: Training labels
        iterations: Number of training iterations
        alpha: Learning rate
        
    Returns:
        Trained model parameters
    """
    
    # Initialize network parameters
    W1, b1, W2, b2 = init_params()
    
    # Training loop
    for i in range(int(iterations)):
        # Forward pass to get activations
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # Backward pass to get gradients
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        
        # Update parameters using gradients
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Print accuracy every 50 iterations to monitor progress
        if i % 50 == 0:
            preds = get_predictions(A2)
            acc = get_accuracy(preds, Y)
            print(f"Epoch {i}: accuracy = {acc:.4f}")
    
    # Return trained parameters
    return W1, b1, W2, b2

# Prediction and visualization functions

def make_predictions(X, W1, b1, W2, b2):
    """
    Make predictions on new data using trained model
    
    Args:
        X: Input features
        W1, b1, W2, b2: Trained parameters
        
    Returns:
        Predicted digit for each example
    """
    
    # Run forward propagation and extract only final output
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    
    # Convert output probabilities to class predictions
    predictions = get_predictions(A2)
    
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    """
    Test model on a specific example and visualize the result
    
    Args:
        index: Index of example to test
        W1, b1, W2, b2: Trained parameters
    """
    # Get the example image (add dimension for single example)
    current_image = X_train[:, index, None]
    
    # Make prediction
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    
    # Get true label
    label = Y_train[index]
    
    # Print prediction and true label
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    # Display the image
    # Reshape from 784×1 to 28×28 and scale back to 0-255 for visualization
    current_image = current_image.reshape((28, 28)) * 255
    
    plt.gray()  # Use grayscale colormap
    
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Train the model using gradient descent
    # 500 iterations, learning rate of 0.10, can be adjusted
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.10)
    
    # Test and visualize predictions on first 4 examples
    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
    test_prediction(2, W1, b1, W2, b2)
    test_prediction(3, W1, b1, W2, b2)
