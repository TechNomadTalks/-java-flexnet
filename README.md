# FlexNet - A Complete Neural Network Implementation in Pure Java

This repository contains a fully functional multi-layer perceptron (MLP) neural network implemented entirely from scratch in Java, without using any external libraries such as ND4J, DeepLearning4J, TensorFlow, or PyTorch.

All functionality including feedforward computation, backpropagation, gradient descent optimization, and model persistence has been implemented using only standard Java libraries (java.util, java.io).

## The Problem

Neural networks and deep learning are fundamental to modern artificial intelligence, yet their inner workings remain a black box for many developers. Understanding how neural networks actually work under the hood requires implementing them from scratch.

This project addresses several key challenges:

**1. Learning Fundamental Concepts**
Implementing a neural network from scratch forces deep understanding of:
- How feedforward computation propagates signals through layers
- How backpropagation uses the chain rule to compute gradients
- How gradient descent updates weights to minimize loss
- How different activation functions affect learning

**2. No Dependencies**
Production deep learning libraries like TensorFlow, PyTorch, and DeepLearning4J are powerful but complex. This implementation shows that the core concepts can be implemented with basic programming knowledge and standard tools.

**3. Educational Value**
This code serves as a learning resource for developers who want to:
- Understand the mathematics behind neural networks
- See how theory translates to working code
- Experiment with different architectures and hyperparameters

## The Solution

FlexNet provides a complete, working neural network implementation that demonstrates:

### Core Implementation

- **Feedforward Computation**: Matrix multiplication with bias addition, followed by non-linear activation functions
- **Backpropagation**: Gradient computation using the chain rule, propagating errors backward through the network
- **Gradient Descent**: Weight updates using learning rate, with options for momentum acceleration and L2 regularization
- **Weight Initialization**: Xavier/Glorot for sigmoid/tanh, He for ReLU - critical for stable training

### Activation Functions

Six activation functions are implemented, each with specific use cases:

- **Sigmoid**: Binary classification output (0 to 1)
- **Tanh**: Hidden layers when zero-centered output is needed (-1 to 1)
- **ReLU**: Fast training, common choice for deep networks
- **LeakyReLU**: Addresses the "dying ReLU" problem
- **ELU**: Smooth activation with negative values
- **Softmax**: Multi-class classification output

### Loss Functions

- **MSE (Mean Squared Error)**: Regression tasks
- **Cross-Entropy**: Classification tasks, works with softmax

### Training Metrics

- Accuracy, precision, recall, F1 score for classification
- Confusion matrix for detailed performance analysis
- Loss tracking throughout training

## What This Demonstrates

This implementation showcases Java programming skills including:

- Object-oriented design with clean class structure
- Mathematical implementation without external libraries
- Careful handling of numerical stability (e.g., sigmoid overflow prevention)
- Comprehensive error handling and input validation
- Model serialization for persistence
- Unit testing methodology

## Features

### Network Architecture

The network supports fully connected (dense) layers with configurable neuron counts:

- Input layer size (number of input features)
- Hidden layers (array of neuron counts for each hidden layer)
- Output layer size (number of output neurons)

### Weight Initialization

Appropriate initialization methods based on activation function:

- Xavier/Glorot initialization for sigmoid and tanh
- He initialization for ReLU
- Biases initialized to zero

### Training Options

- **Learning Rate**: Step size during gradient descent
- **Momentum**: Accelerates convergence
- **Regularization**: L2 penalty to prevent overfitting
- **Batch Size**: Stochastic (1), mini-batch, or full-batch

### Model Persistence

Save and load trained models with all weights and architecture.

## Usage

### Basic Example

```java
// Create a network for XOR: 2 inputs, 4 hidden neurons, 1 output
FlexNet net = new FlexNet(2, new int[]{4}, 1, 
                         new String[]{"tanh", "sigmoid"}, 0.5);

// Define training data
double[][] inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
double[][] targets = {{0.0}, {1.0}, {1.0}, {0.0}};

// Train the network
net.train(inputs, targets, 2000);

// Make predictions
double[] prediction = net.predict(new double[]{0.0, 1.0});
System.out.println("Prediction: " + prediction[0]); // Close to 1.0
```

### Classification Example

```java
// Multi-class classification with cross-entropy loss
FlexNet net = new FlexNet(64, new int[]{128, 64}, 10,
                         new String[]{"relu", "relu", "softmax"}, 0.01, 0.9, 0.001, 32);
net.setLossType(LossType.CROSS_ENTROPY);
net.train(inputs, targets, 100);

// Evaluate
double accuracy = net.computeAccuracy(inputs, labels);
System.out.println("Accuracy: " + accuracy);
```

## Implementation Details

### Feedforward

For each layer: z = W * a + b, then a = activation(z)

### Backpropagation

1. Forward pass: Compute predictions
2. Output layer error: delta = gradient_loss * activation_derivative
3. Backward pass: Propagate errors, compute gradients
4. Update: W = W - learning_rate * gradient

### Loss Functions

**MSE**: (1/n) * sum((predicted - target)^2)

**Cross-Entropy**: -sum(target * log(predicted))

## Running the Demos

```bash
cd neural_network
javac *.java
java Demo          # XOR classification
java MnistDemo    # Digit classification
java RegressionDemo  # Function approximation
java FlexNetTest  # Run unit tests
```

## Files

- **FlexNet.java** - Main neural network implementation
- **Demo.java** - XOR training demonstration
- **MnistDemo.java** - Digit classification demonstration
- **RegressionDemo.java** - Function approximation demonstration
- **FlexNetTest.java** - Comprehensive unit tests

## Requirements

- Java 8 or higher
- No external dependencies

## License

This code is provided as-is for educational purposes.

## Acknowledgments

This implementation was created entirely from scratch to demonstrate understanding of neural network fundamentals. No external libraries or frameworks were used in the development process.
