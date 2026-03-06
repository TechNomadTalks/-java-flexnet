import java.io.*;
import java.util.*;

/**
 * FlexNet - A flexible multi-layer perceptron (fully connected neural network) implementation
 * 
 * This class implements a complete neural network with:
 * - Configurable architecture (input, hidden layers, output)
 * - Multiple activation functions (sigmoid, tanh, relu, linear)
 * - Xavier/Glorot and He weight initialization
 * - Backpropagation with momentum and L2 regularization
 * - Batch, mini-batch, and stochastic gradient descent
 * - Model serialization/deserialization
 * 
 * @author FlexNet Implementation
 * @version 1.0
 */
public class FlexNet {
    
    // Network architecture
    private int inputSize;
    private int[] hiddenLayers;
    private int outputSize;
    
    // Weights and biases
    // weights[layer][neuron][input_index] - weights for layer L going to neuron N from previous layer
    // biases[layer][neuron] - bias for neuron N in layer L
    private double[][][] weights;
    private double[][] biases;
    
    // Velocity for momentum-based gradient descent
    private double[][][] velocityW;
    private double[][] velocityB;
    
    // Activation functions for each layer (including output layer)
    private String[] activations;
    
    // Hyperparameters
    private double learningRate;
    private double momentum;
    private double regularization;  // L2 regularization strength
    private int batchSize;
    
    // Random number generator for weight initialization
    private Random random;
    
    // Store pre-activation (z) and post-activation (a) values for backpropagation
    // preActivations[layer][neuron] = z_l = W_l * a_{l-1} + b_l
    // postActivations[layer][neuron] = a_l = activation(z_l)
    private double[][] preActivations;
    private double[][] postActivations;
    
    // Delta values for backpropagation
    // delta[layer][neuron] = partial derivative of loss with respect to z_l
    private double[][] deltas;
    
    /**
     * Constructor with all parameters
     * 
     * @param inputSize Number of input features
     * @param hiddenLayers Array of neuron counts for hidden layers
     * @param outputSize Number of output neurons
     * @param activations Array of activation functions for each layer (including output)
     * @param learningRate Learning rate for gradient descent
     * @param momentum Momentum coefficient (0.0 for no momentum)
     * @param regularization L2 regularization strength (0.0 for no regularization)
     * @param batchSize Batch size (1 for SGD, n for batch, other for mini-batch)
     */
    public FlexNet(int inputSize, int[] hiddenLayers, int outputSize, 
                   String[] activations, double learningRate, 
                   double momentum, double regularization, int batchSize) {
        
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers != null ? hiddenLayers.clone() : new int[0];
        this.outputSize = outputSize;
        this.activations = activations != null ? activations.clone() : new String[this.hiddenLayers.length + 1];
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.regularization = regularization;
        this.batchSize = batchSize;
        this.random = new Random();
        
        initializeArchitecture();
        initializeWeights();
    }
    
    /**
     * Constructor with default momentum and regularization
     */
    public FlexNet(int inputSize, int[] hiddenLayers, int outputSize, 
                   String[] activations, double learningRate) {
        this(inputSize, hiddenLayers, outputSize, activations, 
             learningRate, 0.0, 0.0, 32);
    }
    
    /**
     * Constructor with default batch size
     */
    public FlexNet(int inputSize, int[] hiddenLayers, int outputSize, 
                   String[] activations, double learningRate, 
                   double momentum, double regularization) {
        this(inputSize, hiddenLayers, outputSize, activations, 
             learningRate, momentum, regularization, 32);
    }
    
    /**
     * Initialize the network architecture arrays
     */
    private void initializeArchitecture() {
        int numLayers = hiddenLayers.length + 1;  // Hidden layers + output layer
        
        // Ensure activations array is the correct size
        if (activations.length != numLayers) {
            String[] newActivations = new String[numLayers];
            Arrays.fill(newActivations, "relu");
            this.activations = newActivations;
        }
        
        // Initialize weights and biases
        // Layer 0 connects input to first hidden layer
        // Layer i connects layer i-1 to layer i
        
        weights = new double[numLayers][][];
        biases = new double[numLayers][];
        velocityW = new double[numLayers][][];
        velocityB = new double[numLayers][];
        
        // First layer: inputSize -> hiddenLayers[0]
        int prevSize = inputSize;
        for (int l = 0; l < numLayers; l++) {
            int currentSize = (l < hiddenLayers.length) ? hiddenLayers[l] : outputSize;
            
            // weights[l][n][i] = weight from input i to neuron n in layer l
            weights[l] = new double[currentSize][prevSize];
            biases[l] = new double[currentSize];
            velocityW[l] = new double[currentSize][prevSize];
            velocityB[l] = new double[currentSize];
            
            prevSize = currentSize;
        }
        
        // Initialize storage for forward/backward pass
        preActivations = new double[numLayers][];
        postActivations = new double[numLayers][];
        deltas = new double[numLayers][];
        
        for (int l = 0; l < numLayers; l++) {
            int size = (l < hiddenLayers.length) ? hiddenLayers[l] : outputSize;
            preActivations[l] = new double[size];
            postActivations[l] = new double[size];
            deltas[l] = new double[size];
        }
    }
    
    /**
     * Initialize weights using appropriate initialization scheme
     * 
     * Xavier/Glorot initialization:
     * - Good for sigmoid and tanh activations
     * - Weights ~ Uniform[-limit, limit] where limit = sqrt(6 / (fanIn + fanOut))
     * - Or Gaussian with mean=0, std = sqrt(2 / (fanIn + fanOut))
     * 
     * He initialization:
     * - Good for ReLU and its variants
     * - Weights ~ Gaussian with mean=0, std = sqrt(2 / fanIn)
     * 
     * @param layerIndex The layer index to initialize
     * @param fanIn Number of inputs to the layer
     * @param fanOut Number of neurons in the layer
     */
    private void initializeWeights() {
        int numLayers = weights.length;
        
        for (int l = 0; l < numLayers; l++) {
            int fanIn = weights[l][0].length;  // Number of inputs
            int fanOut = weights[l].length;    // Number of neurons
            
            String activation = activations[l];
            double limit;
            
            // Determine initialization based on activation function
            if (activation.equalsIgnoreCase("relu") || 
                activation.equalsIgnoreCase("leakyrelu")) {
                // He initialization for ReLU
                // std = sqrt(2 / fanIn)
                double std = Math.sqrt(2.0 / fanIn);
                for (int n = 0; n < weights[l].length; n++) {
                    for (int i = 0; i < weights[l][n].length; i++) {
                        // Gaussian with std
                        weights[l][n][i] = random.nextGaussian() * std;
                    }
                    // Biases initialized to small positive values for ReLU
                    // This helps with dead neurons
                    biases[l][n] = 0.01;
                }
            } else {
                // Xavier/Glorot initialization for sigmoid and tanh
                // limit = sqrt(6 / (fanIn + fanOut))
                limit = Math.sqrt(6.0 / (fanIn + fanOut));
                for (int n = 0; n < weights[l].length; n++) {
                    for (int i = 0; i < weights[l][n].length; i++) {
                        // Uniform distribution [-limit, limit]
                        weights[l][n][i] = (random.nextDouble() * 2 - 1) * limit;
                    }
                    // Biases initialized to zero
                    biases[l][n] = 0.0;
                }
            }
        }
    }
    
    /**
     * Feedforward computation - computes the network output for given input
     * 
     * For each layer l:
     *   z_l = W_l * a_{l-1} + b_l    (pre-activation)
     *   a_l = activation(z_l)       (post-activation)
     * 
     * Where:
     *   - W_l is the weight matrix for layer l
     *   - a_{l-1} is the output from previous layer (or input for l=0)
     *   - b_l is the bias vector for layer l
     *   - activation is the activation function
     * 
     * @param input Input features (size inputSize)
     * @return Output predictions (size outputSize)
     */
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch. Expected " + inputSize + 
                                               ", got " + input.length);
        }
        
        // Copy input to first layer's post-activation
        postActivations[0] = input.clone();
        
        int numLayers = weights.length;
        
        // Forward pass through each layer
        for (int l = 0; l < numLayers; l++) {
            int currentLayerSize = weights[l].length;
            
            // Compute pre-activation: z_l = W_l * a_{l-1} + b_l
            for (int n = 0; n < currentLayerSize; n++) {
                double sum = biases[l][n];
                
                // Dot product: sum = Σ(w_i * a_i)
                double[] prevActivations = (l == 0) ? input : postActivations[l - 1];
                for (int i = 0; i < prevActivations.length; i++) {
                    sum += weights[l][n][i] * prevActivations[i];
                }
                
                preActivations[l][n] = sum;
            }
            
            // Apply activation function
            postActivations[l] = applyActivation(preActivations[l], activations[l]);
        }
        
        // Return output layer predictions
        return postActivations[numLayers - 1].clone();
    }
    
    /**
     * Apply activation function to an array of pre-activation values
     * 
     * Supported activations:
     * - sigmoid: σ(x) = 1 / (1 + e^(-x))
     *           Derivative: σ'(x) = σ(x) * (1 - σ(x))
     * 
     * - tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     *        Derivative: tanh'(x) = 1 - tanh²(x)
     * 
     * - relu: max(0, x)
     *        Derivative: 1 if x > 0, 0 otherwise (or α for leaky ReLU)
     * 
     * - linear: x (identity)
     *          Derivative: 1
     * 
     * @param values Array of pre-activation values
     * @param activationType Type of activation function
     * @return Array of post-activation values
     */
    private double[] applyActivation(double[] values, String activationType) {
        double[] result = new double[values.length];
        
        switch (activationType.toLowerCase()) {
            case "sigmoid":
                for (int i = 0; i < values.length; i++) {
                    // Sigmoid: σ(x) = 1 / (1 + e^(-x))
                    // Numerically stable version: if x < 0, compute as e^x / (1 + e^x)
                    if (values[i] >= 0) {
                        result[i] = 1.0 / (1.0 + Math.exp(-values[i]));
                    } else {
                        double expX = Math.exp(values[i]);
                        result[i] = expX / (1.0 + expX);
                    }
                }
                break;
                
            case "tanh":
                for (int i = 0; i < values.length; i++) {
                    // Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                    result[i] = Math.tanh(values[i]);
                }
                break;
                
            case "relu":
                for (int i = 0; i < values.length; i++) {
                    // ReLU: max(0, x)
                    result[i] = Math.max(0, values[i]);
                }
                break;
                
            case "leakyrelu":
                for (int i = 0; i < values.length; i++) {
                    // Leaky ReLU: x if x > 0, 0.01x otherwise
                    result[i] = (values[i] > 0) ? values[i] : 0.01 * values[i];
                }
                break;
                
            case "linear":
            case "identity":
                // Linear/Identity: f(x) = x
                System.arraycopy(values, 0, result, 0, values.length);
                break;
                
            default:
                throw new IllegalArgumentException("Unknown activation: " + activationType);
        }
        
        return result;
    }
    
    /**
     * Compute derivative of activation function
     * 
     * @param preActivationValue The pre-activation (z) value
     * @param activationType The type of activation function
     * @return The derivative value
     */
    private double activationDerivative(double preActivationValue, String activationType) {
        switch (activationType.toLowerCase()) {
            case "sigmoid":
                // σ'(x) = σ(x) * (1 - σ(x))
                // Using pre-activation: σ'(x) = (1 / (1 + e^(-x))) * (1 - 1 / (1 + e^(-x)))
                double sigmoid;
                if (preActivationValue >= 0) {
                    sigmoid = 1.0 / (1.0 + Math.exp(-preActivationValue));
                } else {
                    double expX = Math.exp(preActivationValue);
                    sigmoid = expX / (1.0 + expX);
                }
                return sigmoid * (1.0 - sigmoid);
                
            case "tanh":
                // tanh'(x) = 1 - tanh²(x)
                double tanhValue = Math.tanh(preActivationValue);
                return 1.0 - tanhValue * tanhValue;
                
            case "relu":
                // ReLU': 1 if x > 0, 0 otherwise
                return (preActivationValue > 0) ? 1.0 : 0.0;
                
            case "leakyrelu":
                // Leaky ReLU': 1 if x > 0, 0.01 otherwise
                return (preActivationValue > 0) ? 1.0 : 0.01;
                
            case "linear":
            case "identity":
                // Linear': 1
                return 1.0;
                
            default:
                throw new IllegalArgumentException("Unknown activation: " + activationType);
        }
    }
    
    /**
     * Train the neural network on a single sample using stochastic gradient descent
     * 
     * Backpropagation algorithm:
     * 1. Forward pass: compute predictions
     * 2. Compute output layer error: δ_L = ∂L/∂a_L * σ'(z_L)
     * 3. Backpropagate errors: δ_l = (W_{l+1}^T * δ_{l+1}) * σ'(z_l)
     * 4. Compute gradients: ∂L/∂W_l = δ_l * a_{l-1}^T, ∂L/∂b_l = δ_l
     * 5. Update weights: W_l = W_l - η * (∂L/∂W_l + λ * W_l) + momentum * v
     * 
     * Where:
     *   - L is the loss function (MSE)
     *   - η is the learning rate
     *   - λ is the regularization strength
     *   - v is the velocity for momentum
     * 
     * @param input Input features
     * @param target Target output
     */
    public void train(double[] input, double[] target) {
        // Forward pass
        predict(input);
        
        int numLayers = weights.length;
        int outputLayer = numLayers - 1;
        
        // Compute output layer delta (error term)
        // For MSE loss: L = (1/n) * Σ(y_true - y_pred)²
        // ∂L/∂a_L = -(2/n) * (y_true - y_pred)
        // δ_L = ∂L/∂a_L * σ'(z_L)
        
        double[] output = postActivations[outputLayer];
        
        for (int n = 0; n < outputSize; n++) {
            double error = output[n] - target[n];
            deltas[outputLayer][n] = error * activationDerivative(
                preActivations[outputLayer][n], activations[outputLayer]);
        }
        
        // Backpropagate through hidden layers
        // δ_l = (W_{l+1}^T * δ_{l+1}) * σ'(z_l)
        for (int l = outputLayer - 1; l >= 0; l--) {
            int nextLayerSize = weights[l + 1].length;
            
            for (int n = 0; n < weights[l].length; n++) {
                double errorSum = 0.0;
                
                // Sum of weighted errors from next layer
                for (int nextN = 0; nextN < nextLayerSize; nextN++) {
                    errorSum += weights[l + 1][nextN][n] * deltas[l + 1][nextN];
                }
                
                deltas[l][n] = errorSum * activationDerivative(
                    preActivations[l][n], activations[l]);
            }
        }
        
        // Update weights and biases
        // Using gradient descent: W = W - learningRate * gradient
        // With momentum: v = momentum * v - learningRate * gradient
        //                W = W + v
        
        double[] prevActivations = input;
        
        for (int l = 0; l < numLayers; l++) {
            if (l > 0) {
                prevActivations = postActivations[l - 1];
            }
            
            for (int n = 0; n < weights[l].length; n++) {
                // Update bias: b = b - η * δ
                double delta = deltas[l][n];
                
                // Apply momentum to bias
                velocityB[l][n] = momentum * velocityB[l][n] - learningRate * delta;
                biases[l][n] += velocityB[l][n];
                
                // Update weights: W = W - η * (δ * a_prev + λ * W)
                for (int i = 0; i < weights[l][n].length; i++) {
                    // Gradient = δ * a_prev + regularization * W
                    double gradient = delta * prevActivations[i] + 
                                     regularization * weights[l][n][i];
                    
                    // Apply momentum to weights
                    velocityW[l][n][i] = momentum * velocityW[l][n][i] - 
                                         learningRate * gradient;
                    weights[l][n][i] += velocityW[l][n][i];
                }
            }
        }
    }
    
    /**
     * Train the neural network on a batch of samples
     * 
     * Performs mini-batch gradient descent:
     * 1. Accumulate gradients over batch
     * 2. Update weights after processing batch
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs
     */
    public void train(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        
        // Stochastic gradient descent (batchSize = 1)
        if (batchSize == 1) {
            for (int i = 0; i < inputs.length; i++) {
                train(inputs[i], targets[i]);
            }
            return;
        }
        
        // Mini-batch gradient descent
        int numSamples = inputs.length;
        int numBatches = (int) Math.ceil((double) numSamples / batchSize);
        
        // Store accumulated gradients
        double[][][] gradW = new double[weights.length][][];
        double[][] gradB = new double[weights.length][];
        
        for (int l = 0; l < weights.length; l++) {
            gradW[l] = new double[weights[l].length][weights[l][0].length];
            gradB[l] = new double[weights[l].length];
        }
        
        // Process each mini-batch
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * batchSize;
            int endIdx = Math.min(startIdx + batchSize, numSamples);
            int batchActualSize = endIdx - startIdx;
            
            // Reset accumulated gradients for this batch
            for (int l = 0; l < weights.length; l++) {
                for (int n = 0; n < weights[l].length; n++) {
                    Arrays.fill(gradW[l][n], 0.0);
                    gradB[l][n] = 0.0;
                }
            }
            
            // Forward and backward pass for each sample in batch
            int numLayers = weights.length;
            int outputLayer = numLayers - 1;
            
            for (int i = startIdx; i < endIdx; i++) {
                // Forward pass
                predict(inputs[i]);
                
                // Compute output layer delta
                double[] output = postActivations[outputLayer];
                
                for (int n = 0; n < outputSize; n++) {
                    double error = output[n] - targets[i][n];
                    deltas[outputLayer][n] = error * activationDerivative(
                        preActivations[outputLayer][n], activations[outputLayer]);
                }
                
                // Backpropagate
                for (int l = outputLayer - 1; l >= 0; l--) {
                    int nextLayerSize = weights[l + 1].length;
                    
                    for (int n = 0; n < weights[l].length; n++) {
                        double errorSum = 0.0;
                        
                        for (int nextN = 0; nextN < nextLayerSize; nextN++) {
                            errorSum += weights[l + 1][nextN][n] * deltas[l + 1][nextN];
                        }
                        
                        deltas[l][n] = errorSum * activationDerivative(
                            preActivations[l][n], activations[l]);
                    }
                }
                
                // Accumulate gradients
                double[] prevActivations = inputs[i];
                
                for (int l = 0; l < numLayers; l++) {
                    if (l > 0) {
                        prevActivations = postActivations[l - 1];
                    }
                    
                    for (int n = 0; n < weights[l].length; n++) {
                        gradB[l][n] += deltas[l][n];
                        
                        for (int i2 = 0; i2 < weights[l][n].length; i2++) {
                            gradW[l][n][i2] += deltas[l][n] * prevActivations[i2];
                        }
                    }
                }
            }
            
            // Average gradients and update weights
            for (int l = 0; l < numLayers; l++) {
                for (int n = 0; n < weights[l].length; n++) {
                    // Average gradient for bias
                    double avgGradB = gradB[l][n] / batchActualSize;
                    
                    // Apply momentum and update bias
                    velocityB[l][n] = momentum * velocityB[l][n] - 
                                     learningRate * avgGradB;
                    biases[l][n] += velocityB[l][n];
                    
                    // Update weights
                    for (int i2 = 0; i2 < weights[l][n].length; i2++) {
                        double avgGradW = gradW[l][n][i2] / batchActualSize +
                                         regularization * weights[l][n][i2];
                        
                        velocityW[l][n][i2] = momentum * velocityW[l][n][i2] - 
                                             learningRate * avgGradW;
                        weights[l][n][i2] += velocityW[l][n][i2];
                    }
                }
            }
        }
    }
    
    /**
     * Compute Mean Squared Error loss
     * 
     * MSE = (1/n) * Σ(y_true - y_pred)²
     * 
     * This measures the average squared difference between predicted and true values.
     * Lower MSE indicates better fit.
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs
     * @return Mean Squared Error
     */
    public double computeMSE(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        double totalError = 0.0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            
            for (int j = 0; j < targets[i].length; j++) {
                double error = prediction[j] - targets[i][j];
                totalError += error * error;
            }
        }
        
        // MSE = (1/n) * Σ(y_true - y_pred)²
        return totalError / inputs.length;
    }
    
    /**
     * Train the network for a specified number of epochs
     * 
     * @param inputs Training inputs
     * @param targets Training targets
     * @param epochs Number of training epochs
     * @return Training history (MSE per epoch)
     */
    public double[] trainEpochs(double[][] inputs, double[][] targets, int epochs) {
        double[] history = new double[epochs];
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data
            int[] indices = new int[inputs.length];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            
            // Fisher-Yates shuffle
            for (int i = indices.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            // Create shuffled arrays
            double[][] shuffledInputs = new double[inputs.length][];
            double[][] shuffledTargets = new double[targets.length][];
            
            for (int i = 0; i < indices.length; i++) {
                shuffledInputs[i] = inputs[indices[i]];
                shuffledTargets[i] = targets[indices[i]];
            }
            
            // Train on shuffled data
            train(shuffledInputs, shuffledTargets);
            
            // Compute MSE for this epoch
            history[epoch] = computeMSE(inputs, targets);
        }
        
        return history;
    }
    
    /**
     * Save the model to a file
     * 
     * File format:
     * - inputSize
     * - hiddenLayers (comma-separated)
     * - outputSize
     * - activations (comma-separated)
     * - learningRate
     * - momentum
     * - regularization
     * - batchSize
     * - weights and biases
     * 
     * @param filePath Path to save the model
     * @throws IOException If file cannot be written
     */
    public void saveModel(String filePath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            // Save architecture
            writer.println(inputSize);
            
            // Save hidden layers
            writer.println(hiddenLayers.length);
            for (int size : hiddenLayers) {
                writer.println(size);
            }
            
            writer.println(outputSize);
            
            // Save activations
            writer.println(activations.length);
            for (String activation : activations) {
                writer.println(activation);
            }
            
            // Save hyperparameters
            writer.println(learningRate);
            writer.println(momentum);
            writer.println(regularization);
            writer.println(batchSize);
            
            // Save weights
            for (int l = 0; l < weights.length; l++) {
                writer.println(weights[l].length);  // Number of neurons
                writer.println(weights[l][0].length);  // Number of inputs
                
                for (int n = 0; n < weights[l].length; n++) {
                    for (int i = 0; i < weights[l][n].length; i++) {
                        writer.println(weights[l][n][i]);
                    }
                }
            }
            
            // Save biases
            for (int l = 0; l < biases.length; l++) {
                writer.println(biases[l].length);
                for (int n = 0; n < biases[l].length; n++) {
                    writer.println(biases[l][n]);
                }
            }
        }
    }
    
    /**
     * Load a model from a file
     * 
     * @param filePath Path to the model file
     * @return Loaded FlexNet model
     * @throws IOException If file cannot be read
     */
    public static FlexNet loadModel(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // Load architecture
            int inputSize = Integer.parseInt(reader.readLine().trim());
            
            int numHidden = Integer.parseInt(reader.readLine().trim());
            int[] hiddenLayers = new int[numHidden];
            for (int i = 0; i < numHidden; i++) {
                hiddenLayers[i] = Integer.parseInt(reader.readLine().trim());
            }
            
            int outputSize = Integer.parseInt(reader.readLine().trim());
            
            // Load activations
            int numActivations = Integer.parseInt(reader.readLine().trim());
            String[] activations = new String[numActivations];
            for (int i = 0; i < numActivations; i++) {
                activations[i] = reader.readLine().trim();
            }
            
            // Load hyperparameters
            double learningRate = Double.parseDouble(reader.readLine().trim());
            double momentum = Double.parseDouble(reader.readLine().trim());
            double regularization = Double.parseDouble(reader.readLine().trim());
            int batchSize = Integer.parseInt(reader.readLine().trim());
            
            // Create network
            FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                          activations, learningRate, momentum,
                                          regularization, batchSize);
            
            // Load weights
            for (int l = 0; l < network.weights.length; l++) {
                int numNeurons = Integer.parseInt(reader.readLine().trim());
                int numInputs = Integer.parseInt(reader.readLine().trim());
                
                for (int n = 0; n < numNeurons; n++) {
                    for (int i = 0; i < numInputs; i++) {
                        network.weights[l][n][i] = Double.parseDouble(reader.readLine().trim());
                    }
                }
            }
            
            // Load biases
            for (int l = 0; l < network.biases.length; l++) {
                int numBiases = Integer.parseInt(reader.readLine().trim());
                for (int n = 0; n < numBiases; n++) {
                    network.biases[l][n] = Double.parseDouble(reader.readLine().trim());
                }
            }
            
            return network;
        }
    }
    
    /**
     * Get the current learning rate
     * 
     * @return Current learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Set the learning rate
     * 
     * The learning rate controls the step size during gradient descent.
     * Typical values: 0.001 to 1.0
     * 
     * @param learningRate New learning rate
     */
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }
    
    /**
     * Get the momentum coefficient
     * 
     * @return Momentum coefficient
     */
    public double getMomentum() {
        return momentum;
    }
    
    /**
     * Set the momentum coefficient
     * 
     * Momentum helps accelerate gradient descent by adding a fraction
     * of the previous gradient update to the current update.
     * Typical values: 0.0 to 0.99
     * 
     * @param momentum New momentum coefficient
     */
    public void setMomentum(double momentum) {
        if (momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException("Momentum must be between 0 and 1");
        }
        this.momentum = momentum;
    }
    
    /**
     * Get the regularization strength
     * 
     * @return L2 regularization strength
     */
    public double getRegularization() {
        return regularization;
    }
    
    /**
     * Set the regularization strength
     * 
     * L2 regularization adds a penalty term to the loss function:
     * L_total = L_original + λ * Σ(w²)
     * This helps prevent overfitting by penalizing large weights.
     * 
     * @param regularization New regularization strength
     */
    public void setRegularization(double regularization) {
        if (regularization < 0) {
            throw new IllegalArgumentException("Regularization must be non-negative");
        }
        this.regularization = regularization;
    }
    
    /**
     * Get the batch size
     * 
     * @return Batch size
     */
    public int getBatchSize() {
        return batchSize;
    }
    
    /**
     * Set the batch size
     * 
     * Batch size determines how many samples are processed before
     * updating weights:
     * - 1: Stochastic Gradient Descent (SGD)
     * - n (full dataset): Batch Gradient Descent
     * - Other: Mini-batch Gradient Descent
     * 
     * @param batchSize New batch size
     */
    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        this.batchSize = batchSize;
    }
    
    /**
     * Get the input size
     * 
     * @return Number of input features
     */
    public int getInputSize() {
        return inputSize;
    }
    
    /**
     * Get the output size
     * 
     * @return Number of output neurons
     */
    public int getOutputSize() {
        return outputSize;
    }
    
    /**
     * Get the hidden layers configuration
     * 
     * @return Array of hidden layer sizes
     */
    public int[] getHiddenLayers() {
        return hiddenLayers.clone();
    }
    
    /**
     * Get the activation functions
     * 
     * @return Array of activation function names
     */
    public String[] getActivations() {
        return activations.clone();
    }
    
    /**
     * Get the number of layers
     * 
     * @return Total number of layers (including input, hidden, and output)
     */
    public int getNumLayers() {
        return hiddenLayers.length + 1;
    }
    
    /**
     * Reset momentum velocities to zero
     * Useful when restarting training
     */
    public void resetMomentum() {
        for (int l = 0; l < velocityW.length; l++) {
            for (int n = 0; n < velocityW[l].length; n++) {
                Arrays.fill(velocityW[l][n], 0.0);
                velocityB[l][n] = 0.0;
            }
        }
    }
    
    /**
     * Get total number of parameters (weights + biases)
     * 
     * @return Total parameter count
     */
    public int getTotalParameters() {
        int total = 0;
        
        for (int l = 0; l < weights.length; l++) {
            total += weights[l].length * weights[l][0].length;  // Weights
            total += biases[l].length;  // Biases
        }
        
        return total;
    }
    
    /**
     * Print network architecture summary
     */
    public void printArchitecture() {
        System.out.println("FlexNet Architecture");
        System.out.println("====================");
        System.out.println("Input Size: " + inputSize);
        System.out.print("Hidden Layers: ");
        for (int size : hiddenLayers) {
            System.out.print(size + " ");
        }
        System.out.println();
        System.out.println("Output Size: " + outputSize);
        System.out.print("Activations: ");
        for (String act : activations) {
            System.out.print(act + " ");
        }
        System.out.println();
        System.out.println("Learning Rate: " + learningRate);
        System.out.println("Momentum: " + momentum);
        System.out.println("Regularization: " + regularization);
        System.out.println("Batch Size: " + batchSize);
        System.out.println("Total Parameters: " + getTotalParameters());
    }
    
    /**
     * Main method for testing and demonstration
     */
    public static void main(String[] args) {
        System.out.println("FlexNet - Neural Network Demonstration");
        System.out.println("========================================\n");
        
        // Create a simple network for XOR problem
        // XOR requires at least one hidden layer with sufficient neurons
        int inputSize = 2;
        int[] hiddenLayers = {4};  // 4 neurons in hidden layer
        int outputSize = 1;
        String[] activations = {"tanh", "sigmoid"};
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                      activations, 0.5, 0.9, 0.001, 32);
        
        network.printArchitecture();
        System.out.println();
        
        // XOR training data
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };
        
        System.out.println("Training XOR problem...");
        
        // Train for multiple epochs
        double[] losses = network.trainEpochs(inputs, targets, 10000);
        
        System.out.println("Final MSE: " + losses[losses.length - 1]);
        System.out.println("\nPredictions after training:");
        
        // Test predictions
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            System.out.printf("Input: [%d, %d] -> Output: %.4f (Target: %.0f)%n",
                            (int)inputs[i][0], (int)inputs[i][1],
                            prediction[0], targets[i][0]);
        }
        
        // Test save/load functionality
        try {
            network.saveModel("FlexNet_xor_model.txt");
            System.out.println("\nModel saved to FlexNet_xor_model.txt");
            
            FlexNet loadedNetwork = FlexNet.loadModel("FlexNet_xor_model.txt");
            System.out.println("Model loaded successfully");
            
            // Verify loaded network produces same results
            double[] pred1 = network.predict(inputs[0]);
            double[] pred2 = loadedNetwork.predict(inputs[0]);
            System.out.printf("Verification: Original: %.4f, Loaded: %.4f%n", 
                            pred1[0], pred2[0]);
            
        } catch (IOException e) {
            System.err.println("Error saving/loading model: " + e.getMessage());
        }
        
        // Demonstrate regression with a simple function
        System.out.println("\n========================================");
        System.out.println("Regression Demonstration: f(x) = sin(x)");
        System.out.println("========================================");
        
        int numSamples = 100;
        double[][] regInputs = new double[numSamples][1];
        double[][] regTargets = new double[numSamples][1];
        
        // Generate training data: x in [-π, π], y = sin(x)
        for (int i = 0; i < numSamples; i++) {
            double x = -Math.PI + (2 * Math.PI * i) / (numSamples - 1);
            regInputs[i][0] = x;
            regTargets[i][0] = Math.sin(x);
        }
        
        // Create network for regression
        FlexNet regNetwork = new FlexNet(1, new int[]{16, 16}, 1,
                                         new String[]{"tanh", "tanh", "linear"},
                                         0.01, 0.0, 0.0001, 16);
        
        System.out.println("Training regression network...");
        
        // Train
        regNetwork.trainEpochs(regInputs, regTargets, 500);
        
        double mse = regNetwork.computeMSE(regInputs, regTargets);
        System.out.println("Final MSE: " + mse);
        
        // Test predictions
        System.out.println("\nTest predictions:");
        double[] testPoints = {-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI};
        for (double x : testPoints) {
            double[] input = {x};
            double[] prediction = regNetwork.predict(input);
            System.out.printf("x: %.3f -> Predicted: %.3f, Actual: %.3f%n",
                            x, prediction[0], Math.sin(x));
        }
    }
}
