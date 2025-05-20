# Neural Network from Scratch for MNIST

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/320px-MnistExamples.png)

## Overview
This project implements a simple neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. No deep learning frameworks are used - the implementation builds all components of a neural network (forward propagation, backpropagation, gradient descent) manually to provide a deeper understanding of neural network mechanics.

## Features
- Pure NumPy implementation - no machine learning libraries
- Two-layer neural network with ReLU and Softmax activations
- Proper weight initialization techniques
- Forward and backward propagation implementation
- Batch gradient descent optimization
- Visualization of predictions

## Requirements
- numpy
- pandas
- matplotplib

## Dataset
The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). The dataset is loaded from CSV files with the following format:
- First column: digit label (0-9)
- Remaining 784 columns: pixel values (0-255)

## Implementation Details

### Network Architecture
- Input layer: 784 neurons (28x28 image flattened)
- Hidden layer: 10 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation (one per digit class)

### Key Components

#### Data Preprocessing
- Normalization of pixel values from [0, 255] to [0, 1]
- Splitting into training and development sets
- Shuffling for better training distribution

#### Weight Initialization
- He initialization scaled by √(1/n_inputs) to prevent vanishing/exploding gradients
- Zero initialization for biases

#### Forward Propagation
- Linear transformation followed by ReLU activation in the hidden layer
- Linear transformation followed by Softmax activation in the output layer

#### Backward Propagation
- Calculation of gradients using the chain rule
- Derivative of cross-entropy loss with softmax activation
- Backpropagation of error through the layers

#### Training
- Batch gradient descent for parameter updates
- Configurable learning rate and iterations
- Periodic accuracy reporting

#### Visualization
- Display of example digits
- Comparison of predictions vs. actual labels

## How to Use

1. Download the MNIST dataset in CSV format
2. Update the file path in the code to point to your downloaded CSV file
3. Run the script to train the model

```python
# When run directly
if __name__ == "__main__":
    # Train the model (modify parameters as needed)
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.10)
    
    # Test predictions on sample images
    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
