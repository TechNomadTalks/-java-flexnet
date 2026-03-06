# FlexNet - A Complete Neural Network Implementation in Pure Java

This repository contains a fully functional multi-layer perceptron (MLP) neural network implemented entirely from scratch in Java, without using any external libraries such as ND4J, DeepLearning4J, TensorFlow, or PyTorch.

All functionality including feedforward computation, backpropagation, gradient descent optimization, and model persistence has been implemented using only standard Java libraries (java.util, java.io).

## Overview

FlexNet is a versatile neural network library that provides:

- Configurable multi-layer architecture with any number of hidden layers
- Multiple activation functions (sigmoid, tanh, ReLU, linear)
- Multiple gradient descent variants (stochastic, mini-batch, batch)
- Momentum-based optimization for faster convergence
- L2 regularization to prevent overfitting
- Model serialization for saving and loading trained networks
- Comprehensive loss tracking and training progress monitoring

## Features

### Network Architecture

The network supports fully connected (dense) layers with configurable neuron counts. You can specify:

- Input layer size (number of input features)
- Hidden layers (array of neuron counts for each hidden layer)
- Output layer size (number of output neurons)

### Activation Functions

Four activation functions are implemented:

- **Sigmoid**: Squashes values between 0 and 1, useful for binary classification
- **Tanh (Hyperbolic Tangent)**: Squashes values between -1 and 1, zero-centered
- **ReLU (Rectified Linear Unit)**: Returns max(0, x), computationally efficient
- **Linear**: Identity function, useful for regression output layers

### Weight Initialization

Appropriate initialization methods are used based on the activation function:

- Xavier/Glorot initialization for sigmoid and tanh activations
- He initialization for ReLU activations
- Biases are initialized to zero

### Training Options

The network supports various training configurations:

- **Learning Rate**: Controls the step size during gradient descent
- **Momentum**: Accelerates convergence by adding velocity to weight updates
- **Regularization**: L2 penalty to prevent overfitting
- **Batch Size**: Configurable for stochastic (1), mini-batch, or full-batch training

### Model Persistence

Trained models can be saved to disk and loaded later for inference. The save format includes:

- Network architecture (layer sizes, activations)
- Training hyperparameters
- All weight matrices and bias vectors

## Implementation Details

### Feedforward Computation

The feedforward pass computes outputs layer by layer:

1. For each layer, compute the pre-activation values: z = W * a + b
2. Apply the activation function to get post-activation values: a = activation(z)
3. Use post-activation values as input to the next layer

### Backpropagation Algorithm

The backpropagation algorithm computes gradients using the chain rule:

1. Forward pass: Compute predictions and store all intermediate values
2. Output layer error: Compute delta using the derivative of the loss function
3. Backward pass: Propagate errors layer by layer, computing gradients at each step
4. Update weights: Apply gradient descent with momentum and regularization

### Loss Function

Mean Squared Error (MSE) is used as the loss function:

MSE = (1/n) * sum((predicted - target)^2)

The gradient of MSE with respect to the output is simply: (predicted - target)

## Usage

### Basic Example

```java
// Create a network for XOR: 2 inputs, 4 hidden neurons, 1 output
FlexNet net = new FlexNet(2, new int[]{4}, 1, 
                         new String[]{"tanh", "sigmoid"}, 0.5);

// Define training data
double[][] inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

double[][] targets = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
};

// Train the network
net.train(inputs, targets, 2000);

// Make predictions
double[] prediction = net.predict(new double[]{0.0, 1.0});
System.out.println("Prediction: " + prediction[0]); // Should be close to 1.0
```

### Advanced Configuration

```java
// Full constructor with all parameters
FlexNet net = new FlexNet(
    2,              // input size
    new int[]{8, 4}, // two hidden layers with 8 and 4 neurons
    1,              // output size
    new String[]{"relu", "relu", "sigmoid"}, // activations for each layer
    0.01,           // learning rate
    0.9,            // momentum
    0.001,          // L2 regularization
    2               // batch size for mini-batch training
);

// Train with early stopping monitoring
net.train(inputs, targets, 5000);

// Check training loss
List<Double> history = net.getTrainingLossHistory();

// Save the trained model
net.saveModel("trained_model.txt");

// Load model for inference
FlexNet loadedNet = FlexNet.loadModel("trained_model.txt");
```

### API Reference

**Constructor:**

- `FlexNet(int inputSize, int[] hiddenLayers, int outputSize, String[] activations, double learningRate)` - Basic constructor
- `FlexNet(int inputSize, int[] hiddenLayers, int outputSize, String[] activations, double learningRate, double momentum, double regularization, int batchSize)` - Full constructor

**Methods:**

- `double[] predict(double[] input)` - Forward pass, returns output array
- `void train(double[][] inputs, double[][] targets, int epochs)` - Train the network
- `double computeLoss(double[][] inputs, double[][] targets)` - Calculate MSE loss
- `void saveModel(String filename)` - Save model to file
- `static FlexNet loadModel(String filename)` - Load model from file
- `void setLearningRate(double lr)` - Dynamically adjust learning rate
- `List<Double> getTrainingLossHistory()` - Get training loss per epoch

## Running the Demo

Navigate to the neural_network directory and compile:

```bash
cd neural_network
javac FlexNet.java Demo.java
java Demo
```

The demo will:

1. Display the XOR truth table
2. Show network architecture
3. Display predictions before training (random outputs)
4. Train the network on XOR data
5. Show loss progression during training
6. Display predictions after training (correct outputs)
7. Demonstrate different configurations
8. Verify save and load functionality

## Files

- **FlexNet.java** - The main neural network implementation (~1100 lines)
- **Demo.java** - Demonstration program showcasing XOR training (~300 lines)

## Requirements

- Java 8 or higher
- No external dependencies

## License

This code is provided as-is for educational and learning purposes.

## Acknowledgments

This implementation was created entirely from scratch to understand the fundamental concepts behind neural networks and deep learning. No external libraries or frameworks were used in the development process.
