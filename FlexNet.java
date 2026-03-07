import java.io.*;
import java.security.SecureRandom;
import java.util.*;

import exceptions.InvalidArchitectureException;
import exceptions.TrainingException;

/**
 * FlexNet - A flexible multi-layer perceptron (fully connected neural network) implementation
 * 
 * <p>This class implements a complete neural network with:
 * <ul>
 *   <li>Configurable architecture (input, hidden layers, output)</li>
 *   <li>Multiple activation functions (sigmoid, tanh, relu, leakyrelu, elu, softmax, linear)</li>
 *   <li>Xavier/Glorot and He weight initialization</li>
 *   <li>Backpropagation with momentum, Adam optimizer, and L2 regularization</li>
 *   <li>Batch, mini-batch, and stochastic gradient descent</li>
 *   <li>Model serialization/deserialization</li>
 *   <li>Multiple loss functions (MSE, Cross-Entropy, MAE, Huber)</li>
 *   <li>Training metrics (accuracy, precision, recall, F1, confusion matrix)</li>
 *   <li>Learning rate scheduling (step, exponential, cosine annealing)</li>
 *   <li>Early stopping and gradient clipping</li>
 * </ul></p>
 * 
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Create a network for XOR: 2 inputs, 4 hidden neurons, 1 output
 * FlexNet net = new FlexNet(2, new int[]{4}, 1, 
 *                          new String[]{"tanh", "sigmoid"}, 0.5);
 * 
 * // Train the network
 * net.train(inputs, targets, 2000);
 * 
 * // Make predictions
 * double[] prediction = net.predict(new double[]{0.0, 1.0});
 * System.out.println("Prediction: " + prediction[0]); // Close to 1.0
 * }</pre>
 * 
 * @author FlexNet Implementation
 * @version 1.2
 * @since 1.0
 */

/**
 * Specifies the loss function used for training the neural network.
 * 
 * <ul>
 *   <li>{@link #MSE} - Mean Squared Error, best for regression tasks</li>
 *   <li>{@link #CROSS_ENTROPY} - Cross-Entropy, best for classification with softmax</li>
 *   <li>{@link #MAE} - Mean Absolute Error, robust to outliers</li>
 *   <li>{@link #HUBER} - Huber loss, combines MSE and MAE benefits</li>
 * </ul>
 */
enum LossType {
    MSE,
    CROSS_ENTROPY,
    MAE,
    HUBER
}

enum LearningRateSchedule {
    NONE,
    STEP,
    EXPONENTIAL,
    COSINE
}

public class FlexNet {
    
    private int inputSize;
    private int[] hiddenLayers;
    private int outputSize;
    
    private double[][][] weights;
    private double[][] biases;
    
    private double[][][] velocityW;
    private double[][] velocityB;
    
    private String[] activations;
    
    private double learningRate;
    private double momentum;
    private double regularization;
    private int batchSize;
    private double gradientClipThreshold;
    private double huberDelta;
    
    // Adam optimizer
    private boolean useAdam;
    private double adamBeta1;
    private double adamBeta2;
    private double adamEpsilon;
    private int adamT;
    private double[][][] mW;
    private double[][] mB;
    private double[][][] vW;
    private double[][] vB;
    
    private LearningRateSchedule lrSchedule;
    private double initialLearningRate;
    private int stepSize;
    private double stepGamma;
    private double exponentialDecay;
    private int cosinePeriod;
    
    // Early stopping
    private boolean earlyStoppingEnabled;
    private int earlyStoppingPatience;
    private double earlyStoppingMinDelta;
    private double bestValidationLoss;
    private int epochsWithoutImprovement;
    private boolean earlyStoppingTriggered;
    private double[][][] bestWeights;
    private double[][] bestBiases;
    
    private SecureRandom random;
    
    private double[][] preActivations;
    private double[][] postActivations;
    
    private double[][] deltas;
    
    private LossType lossType;
    
    private static final double ELU_ALPHA = 1.0;
    
    private double dropoutRate;
    private boolean dropoutEnabled;
    private double[][] dropoutMasks;
    private Random dropoutRandom;
    
    /**
     * Creates a new FlexNet neural network with the specified architecture.
     * 
     * <p>This constructor creates a fully-connected (dense) multi-layer perceptron
     * with configurable hidden layers and activation functions. The network uses
     * appropriate weight initialization (Xavier/Glorot for sigmoid/tanh, He for ReLU)
     * based on the specified activation functions.</p>
     * 
     * <h3>Supported Activation Functions</h3>
     * <ul>
     *   <li>{@code "sigmoid"} - Sigmoid activation, good for binary output</li>
     *   <li>{@code "tanh"} - Hyperbolic tangent, zero-centered output</li>
     *   <li>{@code "relu"} - Rectified Linear Unit, default for deep networks</li>
     *   <li>{@code "leakyrelu"} - Leaky ReLU, prevents dying ReLU problem</li>
     *   <li>{@code "elu"} - Exponential Linear Unit, smooth activation</li>
     *   <li>{@code "softmax"} - Softmax, for multi-class classification output</li>
     *   <li>{@code "linear"} - Identity function, for regression output</li>
     * </ul>
     * 
     * <h3>Example Usage</h3>
     * <pre>{@code
     * // Create a network for XOR: 2 inputs, 4 hidden neurons, 1 output
     * FlexNet net = new FlexNet(2, new int[]{4}, 1, 
     *                          new String[]{"tanh", "sigmoid"}, 0.5);
     * 
     * // Train on XOR data
     * net.train(inputs, targets, 2000);
     * 
     * // Make predictions
     * double[] prediction = net.predict(new double[]{0.0, 1.0});
     * }</pre>
     * 
     * @param inputSize      Number of input features (must be positive)
     * @param hiddenLayers   Array specifying neuron counts for each hidden layer
     *                       (empty array for no hidden layers)
     * @param outputSize     Number of output neurons (must be positive)
     * @param activations    Array of activation functions for each layer,
     *                       including output layer
     * @param learningRate   Learning rate for gradient descent (must be positive)
     * @param momentum       Momentum coefficient for SGD (0.0 to 1.0)
     * @param regularization L2 regularization strength (0.0 for no regularization)
     * @param batchSize      Mini-batch size (1 for stochastic, N for mini-batch, 
     *                       or all data for full-batch)
     * @throws IllegalArgumentException if inputSize or outputSize is not positive,
     *         or if learningRate or batchSize is not positive
     */
    public FlexNet(int inputSize, int[] hiddenLayers, int outputSize, 
                   String[] activations, double learningRate, 
                   double momentum, double regularization, int batchSize) {
        
        if (learningRate <= 0) {
            throw new InvalidArchitectureException(
                "Learning rate must be positive",
                "learningRate=" + learningRate);
        }
        
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers != null ? hiddenLayers.clone() : new int[0];
        this.outputSize = outputSize;
        this.activations = activations != null ? activations.clone() : new String[this.hiddenLayers.length + 1];
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.regularization = regularization;
        this.batchSize = batchSize;
        this.gradientClipThreshold = 0.0;
        this.huberDelta = 1.0;
        this.lrSchedule = LearningRateSchedule.NONE;
        this.initialLearningRate = learningRate;
        this.stepSize = 100;
        this.stepGamma = 0.1;
        this.exponentialDecay = 0.95;
        this.cosinePeriod = 1000;
        this.earlyStoppingEnabled = false;
        this.earlyStoppingPatience = 10;
        this.earlyStoppingMinDelta = 0.0;
        this.bestValidationLoss = Double.MAX_VALUE;
        this.epochsWithoutImprovement = 0;
        this.earlyStoppingTriggered = false;
        this.useAdam = false;
        this.adamBeta1 = 0.9;
        this.adamBeta2 = 0.999;
        this.adamEpsilon = 1e-8;
        this.adamT = 0;
        this.random = new SecureRandom();
        this.dropoutRate = 0.5;
        this.dropoutEnabled = false;
        this.dropoutRandom = new Random();
        this.lossType = LossType.MSE;
        
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
    
    private void initializeArchitecture() {
        int numLayers = hiddenLayers.length + 1;
        
        if (activations.length != numLayers) {
            String[] newActivations = new String[numLayers];
            Arrays.fill(newActivations, "relu");
            this.activations = newActivations;
        }
        
        weights = new double[numLayers][][];
        biases = new double[numLayers][];
        velocityW = new double[numLayers][][];
        velocityB = new double[numLayers][];
        
        int prevSize = inputSize;
        for (int l = 0; l < numLayers; l++) {
            int currentSize = (l < hiddenLayers.length) ? hiddenLayers[l] : outputSize;
            
            weights[l] = new double[currentSize][prevSize];
            biases[l] = new double[currentSize];
            velocityW[l] = new double[currentSize][prevSize];
            velocityB[l] = new double[currentSize];
            
            prevSize = currentSize;
        }
        
        preActivations = new double[numLayers][];
        postActivations = new double[numLayers][];
        deltas = new double[numLayers][];
        
        for (int l = 0; l < numLayers; l++) {
            int size = (l < hiddenLayers.length) ? hiddenLayers[l] : outputSize;
            preActivations[l] = new double[size];
            postActivations[l] = new double[size];
            deltas[l] = new double[size];
        }
        
        initializeDropoutMasks();
    }
    
    private void initializeDropoutMasks() {
        int numLayers = weights.length;
        dropoutMasks = new double[numLayers - 1][];
        for (int l = 0; l < numLayers - 1; l++) {
            int layerSize = weights[l].length;
            dropoutMasks[l] = new double[layerSize];
        }
    }
    
    /**
     * Initialize weights using Xavier/Glorot or He initialization
     * 
     * Xavier/Glorot: Good for sigmoid and tanh - weights ~ Uniform[-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut))]
     * He: Good for ReLU and variants - weights ~ Gaussian with std = sqrt(2/fanIn)
     */
    private void initializeWeights() {
        int numLayers = weights.length;
        
        for (int l = 0; l < numLayers; l++) {
            int fanIn = weights[l][0].length;
            int fanOut = weights[l].length;
            
            String activation = activations[l];
            double limit;
            
            if (activation.equalsIgnoreCase("relu") || 
                activation.equalsIgnoreCase("leakyrelu") ||
                activation.equalsIgnoreCase("elu")) {
                double std = Math.sqrt(2.0 / fanIn);
                for (int n = 0; n < weights[l].length; n++) {
                    for (int i = 0; i < weights[l][n].length; i++) {
                        weights[l][n][i] = random.nextGaussian() * std;
                    }
                    biases[l][n] = 0.01;
                }
            } else {
                limit = Math.sqrt(6.0 / (fanIn + fanOut));
                for (int n = 0; n < weights[l].length; n++) {
                    for (int i = 0; i < weights[l][n].length; i++) {
                        weights[l][n][i] = (random.nextDouble() * 2 - 1) * limit;
                    }
                    biases[l][n] = 0.0;
                }
            }
        }
    }
    
    /**
     * Performs inference on a single input sample.
     * 
     * <p>This method runs a complete forward pass through all network layers.
     * It applies the specified activation functions at each layer and returns
     * the final output predictions.</p>
     * 
     * <p>For each layer l:</p>
     * <ul>
     *   <li>z_l = W_l * a_{l-1} + b_l (pre-activation)</li>
     *   <li>a_l = activation(z_l) (post-activation)</li>
     * </ul>
     * 
     * <p>Note: This method modifies internal state (preActivations, postActivations)
     * which is used by training methods. Do not call predict() concurrently from
     * multiple threads without proper synchronization.</p>
     * 
     * @param input Array of input features with size equal to inputSize
     * @return Array of output predictions with size equal to outputSize
     * @throws IllegalArgumentException if input.length != inputSize
     * @throws InvalidArchitectureException if input dimensions are invalid
     */
    public double[] predict(double[] input) {
        if (input == null) {
            throw new InvalidArchitectureException(
                "Input cannot be null",
                "input is null");
        }
        
        if (input.length != inputSize) {
            throw new InvalidArchitectureException(
                "Input size mismatch. Expected " + inputSize + ", got " + input.length,
                "inputSize=" + inputSize + ", provided=" + input.length);
        }
        
        postActivations[0] = input.clone();
        
        int numLayers = weights.length;
        
        for (int l = 0; l < numLayers; l++) {
            int currentLayerSize = weights[l].length;
            
            for (int n = 0; n < currentLayerSize; n++) {
                double sum = biases[l][n];
                
                double[] prevActivations = (l == 0) ? input : postActivations[l - 1];
                for (int i = 0; i < prevActivations.length; i++) {
                    sum += weights[l][n][i] * prevActivations[i];
                }
                
                preActivations[l][n] = sum;
            }
            
            postActivations[l] = applyActivation(preActivations[l], activations[l]);
            
            if (dropoutEnabled && l < numLayers - 1) {
                applyDropout(l);
            }
        }
        
        return postActivations[numLayers - 1].clone();
    }
    
    /**
     * Apply activation function to an array of pre-activation values
     * 
     * Supported activations:
     * - sigmoid: σ(x) = 1 / (1 + e^(-x)), derivative: σ'(x) = σ(x) * (1 - σ(x))
     * - tanh: tanh(x), derivative: tanh'(x) = 1 - tanh²(x)
     * - relu: max(0, x), derivative: 1 if x > 0, 0 otherwise
     * - leakyrelu: x if x > 0, 0.01x otherwise
     * - elu: x > 0 ? x : alpha * (e^x - 1)
     * - softmax: exp(x_i) / sum(exp(x_j))
     * - linear: x, derivative: 1
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
                    result[i] = Math.tanh(values[i]);
                }
                break;
                
            case "relu":
                for (int i = 0; i < values.length; i++) {
                    result[i] = Math.max(0, values[i]);
                }
                break;
                
            case "leakyrelu":
                for (int i = 0; i < values.length; i++) {
                    result[i] = (values[i] > 0) ? values[i] : 0.01 * values[i];
                }
                break;
                
            case "elu":
                for (int i = 0; i < values.length; i++) {
                    result[i] = (values[i] > 0) ? values[i] : ELU_ALPHA * (Math.exp(values[i]) - 1);
                }
                break;
                
            case "softmax":
                double max = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < values.length; i++) {
                    if (values[i] > max) {
                        max = values[i];
                    }
                }
                
                double sum = 0.0;
                for (int i = 0; i < values.length; i++) {
                    result[i] = Math.exp(values[i] - max);
                    sum += result[i];
                }
                
                if (sum == 0.0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
                    double defaultValue = 1.0 / values.length;
                    Arrays.fill(result, defaultValue);
                } else {
                    for (int i = 0; i < values.length; i++) {
                        result[i] /= sum;
                    }
                }
                break;
                
            case "linear":
            case "identity":
                System.arraycopy(values, 0, result, 0, values.length);
                break;
                
            default:
                throw new IllegalArgumentException("Unknown activation: " + activationType);
        }
        
        return result;
    }
    
    private void applyDropout(int layer) {
        double scale = 1.0 / (1.0 - dropoutRate);
        for (int n = 0; n < dropoutMasks[layer].length; n++) {
            if (dropoutRandom.nextDouble() < dropoutRate) {
                dropoutMasks[layer][n] = 0.0;
                postActivations[layer][n] = 0.0;
            } else {
                dropoutMasks[layer][n] = scale;
                postActivations[layer][n] *= scale;
            }
        }
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
                double sigmoid;
                if (preActivationValue >= 0) {
                    sigmoid = 1.0 / (1.0 + Math.exp(-preActivationValue));
                } else {
                    double expX = Math.exp(preActivationValue);
                    sigmoid = expX / (1.0 + expX);
                }
                return sigmoid * (1.0 - sigmoid);
                
            case "tanh":
                double tanhValue = Math.tanh(preActivationValue);
                return 1.0 - tanhValue * tanhValue;
                
            case "relu":
                return (preActivationValue > 0) ? 1.0 : 0.0;
                
            case "leakyrelu":
                return (preActivationValue > 0) ? 1.0 : 0.01;
                
            case "elu":
                return (preActivationValue > 0) ? 1.0 : ELU_ALPHA * Math.exp(preActivationValue);
                
            case "linear":
            case "identity":
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
     * @param input Input features
     * @param target Target output
     */
    public void train(double[] input, double[] target) {
        train(input, target, lossType);
    }
    
    /**
     * Train the neural network on a single sample using stochastic gradient descent
     * with specified loss type
     * 
     * @param input Input features
     * @param target Target output
     * @param lossType The loss function to use
     */
    public void train(double[] input, double[] target, LossType lossType) {
        predict(input);
        
        int numLayers = weights.length;
        int outputLayer = numLayers - 1;
        
        double[] output = postActivations[outputLayer];
        
        if (lossType == LossType.CROSS_ENTROPY && 
            activations[outputLayer].equalsIgnoreCase("softmax")) {
            for (int n = 0; n < outputSize; n++) {
                deltas[outputLayer][n] = output[n] - target[n];
            }
        } else if (lossType == LossType.MAE) {
            for (int n = 0; n < outputSize; n++) {
                double error = output[n] - target[n];
                double sign = (error > 0) ? 1.0 : (error < 0) ? -1.0 : 0.0;
                deltas[outputLayer][n] = sign * activationDerivative(
                    preActivations[outputLayer][n], activations[outputLayer]);
            }
        } else if (lossType == LossType.HUBER) {
            for (int n = 0; n < outputSize; n++) {
                double error = output[n] - target[n];
                double absError = Math.abs(error);
                double gradient;
                if (absError <= huberDelta) {
                    gradient = error;
                } else {
                    gradient = huberDelta * ((error > 0) ? 1.0 : -1.0);
                }
                deltas[outputLayer][n] = gradient * activationDerivative(
                    preActivations[outputLayer][n], activations[outputLayer]);
            }
        } else if (lossType == LossType.CROSS_ENTROPY) {
            for (int n = 0; n < outputSize; n++) {
                double error = output[n] - target[n];
                deltas[outputLayer][n] = error * activationDerivative(
                    preActivations[outputLayer][n], activations[outputLayer]);
            }
        } else {
            for (int n = 0; n < outputSize; n++) {
                double error = output[n] - target[n];
                deltas[outputLayer][n] = error * activationDerivative(
                    preActivations[outputLayer][n], activations[outputLayer]);
            }
        }
        
        for (int l = outputLayer - 1; l >= 0; l--) {
            int nextLayerSize = weights[l + 1].length;
            
            for (int n = 0; n < weights[l].length; n++) {
                double errorSum = 0.0;
                
                for (int nextN = 0; nextN < nextLayerSize; nextN++) {
                    errorSum += weights[l + 1][nextN][n] * deltas[l + 1][nextN];
                }
                
                deltas[l][n] = errorSum * activationDerivative(
                    preActivations[l][n], activations[l]);
                
                if (dropoutEnabled && l < numLayers - 1) {
                    deltas[l][n] *= dropoutMasks[l][n];
                }
            }
        }
        
        double[] prevActivations = input;
        
        for (int l = 0; l < numLayers; l++) {
            if (l > 0) {
                prevActivations = postActivations[l - 1];
            }
            
            for (int n = 0; n < weights[l].length; n++) {
                double delta = deltas[l][n];
                
                if (useAdam) {
                    // Adam update for bias
                    mB[l][n] = adamBeta1 * mB[l][n] + (1 - adamBeta1) * delta;
                    vB[l][n] = adamBeta2 * vB[l][n] + (1 - adamBeta2) * delta * delta;
                    
                    double mHat = mB[l][n] / (1 - Math.pow(adamBeta1, adamT + 1));
                    double vHat = vB[l][n] / (1 - Math.pow(adamBeta2, adamT + 1));
                    
                    if (Double.isNaN(vHat) || Double.isInfinite(vHat)) {
                        vHat = adamEpsilon;
                    }
                    
                    biases[l][n] -= learningRate * mHat / (Math.sqrt(vHat) + adamEpsilon);
                    
                    // Adam update for weights
                    for (int i = 0; i < weights[l][n].length; i++) {
                        double gradient = delta * prevActivations[i];
                        
                        mW[l][n][i] = adamBeta1 * mW[l][n][i] + (1 - adamBeta1) * gradient;
                        vW[l][n][i] = adamBeta2 * vW[l][n][i] + (1 - adamBeta2) * gradient * gradient;
                        
                        double mHatW = mW[l][n][i] / (1 - Math.pow(adamBeta1, adamT + 1));
                        double vHatW = vW[l][n][i] / (1 - Math.pow(adamBeta2, adamT + 1));
                        
                        double regGradient = regularization * weights[l][n][i];
                        weights[l][n][i] -= learningRate * (mHatW / (Math.sqrt(vHatW) + adamEpsilon) + regGradient);
                    }
                } else {
                    // Standard momentum update
                    velocityB[l][n] = momentum * velocityB[l][n] - learningRate * delta;
                    biases[l][n] += velocityB[l][n];
                    
                    for (int i = 0; i < weights[l][n].length; i++) {
                        double gradient = delta * prevActivations[i] + 
                                         regularization * weights[l][n][i];
                        
                        velocityW[l][n][i] = momentum * velocityW[l][n][i] - 
                                             learningRate * gradient;
                        weights[l][n][i] += velocityW[l][n][i];
                    }
                }
            }
        }
        
        if (useAdam) {
            adamT++;
        }
    }
    
    /**
     * Train the neural network on a batch of samples
     * 
     * Performs mini-batch gradient descent: accumulate gradients over batch, then update weights
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs
     */
    public void train(double[][] inputs, double[][] targets) {
        train(inputs, targets, lossType);
    }
    
    /**
     * Train the neural network on a batch of samples with specified loss type
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs
     * @param lossType The loss function to use
     */
    public void train(double[][] inputs, double[][] targets, LossType lossType) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }
        
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] == null) {
                throw new IllegalArgumentException("Input at index " + i + " is null");
            }
            if (inputs[i].length != inputSize) {
                throw new IllegalArgumentException("Input at index " + i + " has wrong size: " + inputs[i].length + " (expected " + inputSize + ")");
            }
            if (targets[i] == null) {
                throw new IllegalArgumentException("Target at index " + i + " is null");
            }
            if (targets[i].length != outputSize) {
                throw new IllegalArgumentException("Target at index " + i + " has wrong size: " + targets[i].length + " (expected " + outputSize + ")");
            }
        }
        
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        
        if (batchSize == 1) {
            for (int i = 0; i < inputs.length; i++) {
                train(inputs[i], targets[i], lossType);
            }
            return;
        }
        
        int numSamples = inputs.length;
        int numBatches = (int) Math.ceil((double) numSamples / batchSize);
        
        double[][][] gradW = new double[weights.length][][];
        double[][] gradB = new double[weights.length][];
        
        for (int l = 0; l < weights.length; l++) {
            gradW[l] = new double[weights[l].length][weights[l][0].length];
            gradB[l] = new double[weights[l].length];
        }
        
        for (int batch = 0; batch < numBatches; batch++) {
            int startIdx = batch * batchSize;
            int endIdx = Math.min(startIdx + batchSize, numSamples);
            int batchActualSize = endIdx - startIdx;
            
            if (batchActualSize <= 0) {
                continue;
            }
            
            for (int l = 0; l < weights.length; l++) {
                for (int n = 0; n < weights[l].length; n++) {
                    Arrays.fill(gradW[l][n], 0.0);
                    gradB[l][n] = 0.0;
                }
            }
            
            int numLayers = weights.length;
            int outputLayer = numLayers - 1;
            
            for (int i = startIdx; i < endIdx; i++) {
                predict(inputs[i]);
                
                double[] output = postActivations[outputLayer];
                
                if (lossType == LossType.CROSS_ENTROPY && 
                    activations[outputLayer].equalsIgnoreCase("softmax")) {
                    for (int n = 0; n < outputSize; n++) {
                        deltas[outputLayer][n] = output[n] - targets[i][n];
                    }
                } else if (lossType == LossType.MAE) {
                    for (int n = 0; n < outputSize; n++) {
                        double error = output[n] - targets[i][n];
                        double sign = (error > 0) ? 1.0 : (error < 0) ? -1.0 : 0.0;
                        deltas[outputLayer][n] = sign * activationDerivative(
                            preActivations[outputLayer][n], activations[outputLayer]);
                    }
                } else if (lossType == LossType.HUBER) {
                    for (int n = 0; n < outputSize; n++) {
                        double error = output[n] - targets[i][n];
                        double absError = Math.abs(error);
                        double gradient;
                        if (absError <= huberDelta) {
                            gradient = error;
                        } else {
                            gradient = huberDelta * ((error > 0) ? 1.0 : -1.0);
                        }
                        deltas[outputLayer][n] = gradient * activationDerivative(
                            preActivations[outputLayer][n], activations[outputLayer]);
                    }
                } else if (lossType == LossType.CROSS_ENTROPY) {
                    for (int n = 0; n < outputSize; n++) {
                        double error = output[n] - targets[i][n];
                        deltas[outputLayer][n] = error * activationDerivative(
                            preActivations[outputLayer][n], activations[outputLayer]);
                    }
                } else {
                    for (int n = 0; n < outputSize; n++) {
                        double error = output[n] - targets[i][n];
                        deltas[outputLayer][n] = error * activationDerivative(
                            preActivations[outputLayer][n], activations[outputLayer]);
                    }
                }
                
                for (int l = outputLayer - 1; l >= 0; l--) {
                    int nextLayerSize = weights[l + 1].length;
                    
                    for (int n = 0; n < weights[l].length; n++) {
                        double errorSum = 0.0;
                        
                        for (int nextN = 0; nextN < nextLayerSize; nextN++) {
                            errorSum += weights[l + 1][nextN][n] * deltas[l + 1][nextN];
                        }
                        
                        deltas[l][n] = errorSum * activationDerivative(
                            preActivations[l][n], activations[l]);
                        
                        if (dropoutEnabled && l < numLayers - 1) {
                            deltas[l][n] *= dropoutMasks[l][n];
                        }
                    }
                }
                
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
            
            // Update weights after accumulating gradients
            double scale = 1.0 / batchActualSize;
            adamT++;
            
            for (int l = 0; l < numLayers; l++) {
                for (int n = 0; n < weights[l].length; n++) {
                    double avgGradB = gradB[l][n] * scale;
                    
                    if (useAdam) {
                        // Adam update for bias
                        mB[l][n] = adamBeta1 * mB[l][n] + (1 - adamBeta1) * avgGradB;
                        vB[l][n] = adamBeta2 * vB[l][n] + (1 - adamBeta2) * avgGradB * avgGradB;
                        
                        double mHat = mB[l][n] / (1 - Math.pow(adamBeta1, adamT));
                        double vHat = vB[l][n] / (1 - Math.pow(adamBeta2, adamT));
                        
                        if (Double.isNaN(vHat) || Double.isInfinite(vHat)) {
                            vHat = adamEpsilon;
                        }
                        
                        biases[l][n] -= learningRate * mHat / (Math.sqrt(vHat) + adamEpsilon);
                        
                        // Adam update for weights
                        for (int i2 = 0; i2 < weights[l][n].length; i2++) {
                            double avgGradW = gradW[l][n][i2] * scale;
                            
                            mW[l][n][i2] = adamBeta1 * mW[l][n][i2] + (1 - adamBeta1) * avgGradW;
                            vW[l][n][i2] = adamBeta2 * vW[l][n][i2] + (1 - adamBeta2) * avgGradW * avgGradW;
                            
                            double mHatW = mW[l][n][i2] / (1 - Math.pow(adamBeta1, adamT));
                            double vHatW = vW[l][n][i2] / (1 - Math.pow(adamBeta2, adamT));
                            
                            weights[l][n][i2] -= learningRate * mHatW / (Math.sqrt(vHatW) + adamEpsilon);
                        }
                    } else {
                        // Standard momentum update
                        if (gradientClipThreshold > 0) {
                            avgGradB = clipGradient(avgGradB);
                        }
                        velocityB[l][n] = momentum * velocityB[l][n] - learningRate * avgGradB;
                        biases[l][n] += velocityB[l][n];
                        
                        for (int i2 = 0; i2 < weights[l][n].length; i2++) {
                            double avgGradW = gradW[l][n][i2] * scale + regularization * weights[l][n][i2];
                            
                            if (gradientClipThreshold > 0) {
                                avgGradW = clipGradient(avgGradW);
                            }
                            
                            velocityW[l][n][i2] = momentum * velocityW[l][n][i2] - learningRate * avgGradW;
                            weights[l][n][i2] += velocityW[l][n][i2];
                        }
                    }
                }
            }
        }
    }
    
    private void initializeAdam() {
        mW = new double[weights.length][][];
        mB = new double[biases.length][];
        vW = new double[weights.length][][];
        vB = new double[biases.length][];
        
        for (int l = 0; l < weights.length; l++) {
            mW[l] = new double[weights[l].length][];
            mB[l] = new double[biases[l].length];
            vW[l] = new double[weights[l].length][];
            vB[l] = new double[biases[l].length];
            
            for (int n = 0; n < weights[l].length; n++) {
                mW[l][n] = new double[weights[l][n].length];
                vW[l][n] = new double[weights[l][n].length];
            }
            Arrays.fill(mB[l], 0.0);
            Arrays.fill(vB[l], 0.0);
        }
        adamT = 0;
    }
    
    /**
     * Clip gradient to prevent exploding gradients
     * 
     * @param gradient The gradient value
     * @return Clipped gradient
     */
    private double clipGradient(double gradient) {
        if (gradient > gradientClipThreshold) {
            return gradientClipThreshold;
        } else if (gradient < -gradientClipThreshold) {
            return -gradientClipThreshold;
        }
        return gradient;
    }
    
    /**
     * Computes the Mean Squared Error (MSE) loss on a dataset.
     * 
     * <p>MSE is commonly used for regression tasks:</p>
     * <pre>MSE = (1/n) * Σ(y_true - y_pred)²</pre>
     * 
     * @param inputs  Input samples with shape [numSamples, inputSize]
     * @param targets True target values with shape [numSamples, outputSize]
     * @return Mean Squared Error
     * @throws IllegalArgumentException if inputs.length != targets.length
     */
    public double computeMSE(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }
        
        double totalError = 0.0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            
            for (int j = 0; j < targets[i].length; j++) {
                double error = prediction[j] - targets[i][j];
                totalError += error * error;
            }
        }
        
        return totalError / inputs.length;
    }
    
    /**
     * Compute Mean Absolute Error loss
     * 
     * MAE = (1/n) * Σ|y_true - y_pred|
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs
     * @return Mean Absolute Error
     */
    public double computeMAE(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }
        
        double totalError = 0.0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            
            for (int j = 0; j < targets[i].length; j++) {
                totalError += Math.abs(prediction[j] - targets[i][j]);
            }
        }
        
        return totalError / inputs.length;
    }
    
    /**
     * Compute Huber loss
     * 
     * Huber loss is quadratic for small errors, linear for large errors
     * L = 0.5 * error^2 for |error| <= delta
     * L = delta * |error| - 0.5 * delta^2 for |error| > delta
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs
     * @return Huber loss
     */
    public double computeHuber(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }
        
        double totalError = 0.0;
        double delta = huberDelta;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            
            for (int j = 0; j < targets[i].length; j++) {
                double error = prediction[j] - targets[i][j];
                double absError = Math.abs(error);
                
                if (absError <= delta) {
                    totalError += 0.5 * error * error;
                } else {
                    totalError += delta * absError - 0.5 * delta * delta;
                }
            }
        }
        
        return totalError / inputs.length;
    }
    
    /**
     * Computes the Cross-Entropy loss on a dataset.
     * 
     * <p>Cross-Entropy is used for classification tasks, especially with softmax:<pre>L = -</p>
     * Σ(y_true * log(y_pred))</pre>
     * 
     * <p>An epsilon value is used to prevent log(0) numerical instability.</p>
     * 
     * @param inputs  Input samples
     * @param targets True target values (one-hot encoded for multi-class)
     * @return Cross-Entropy loss
     * @throws IllegalArgumentException if inputs.length != targets.length
     */
    public double computeCrossEntropy(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
        }
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }
        
        double totalLoss = 0.0;
        double epsilon = 1e-15;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            
            for (int j = 0; j < targets[i].length; j++) {
                double pred = Math.max(epsilon, Math.min(1 - epsilon, prediction[j]));
                totalLoss -= targets[i][j] * Math.log(pred);
            }
        }
        
        return totalLoss / inputs.length;
    }
    
    /**
     * Computes classification accuracy on a dataset.
     * 
     * <p>For multi-class classification, the predicted class is the index with
     * the highest probability (argmax). The accuracy is the fraction of correctly
     * classified samples.</p>
     * 
     * @param inputs Input samples with shape [numSamples, inputSize]
     * @param labels True class indices with shape [numSamples, 1]
     * @return Accuracy as a value between 0.0 and 1.0
     * @throws IllegalArgumentException if inputs.length != labels.length
     */
    public double computeAccuracy(double[][] inputs, int[][] labels) {
        if (inputs.length != labels.length) {
            throw new IllegalArgumentException("Number of inputs and labels must match");
        }
        
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }
        
        int correct = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            if (labels[i] == null || labels[i].length == 0) {
                throw new IllegalArgumentException("Label at index " + i + " is null or empty");
            }
            
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
            if (trueClass < 0 || trueClass >= outputSize) {
                throw new IllegalArgumentException("Label at index " + i + " is out of bounds: " + trueClass + " (expected 0-" + (outputSize - 1) + ")");
            }
            
            if (predictedClass == trueClass) {
                correct++;
            }
        }
        
        return (double) correct / inputs.length;
    }
    
    /**
     * Compute precision for a specific class
     * 
     * Precision = TP / (TP + FP)
     * 
     * @param inputs Array of input samples
     * @param labels Array of true labels (class indices)
     * @param classIndex The class index to compute precision for
     * @return Precision (0.0 to 1.0)
     */
    public double computePrecision(double[][] inputs, int[][] labels, int classIndex) {
        if (inputs.length != labels.length) {
            throw new IllegalArgumentException("Number of inputs and labels must match");
        }
        
        int truePositives = 0;
        int falsePositives = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            if (labels[i] == null || labels[i].length == 0) {
                throw new IllegalArgumentException("Label at index " + i + " is null or empty");
            }
            
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
            if (trueClass < 0 || trueClass >= outputSize) {
                throw new IllegalArgumentException("Label at index " + i + " is out of bounds: " + trueClass + " (expected 0-" + (outputSize - 1) + ")");
            }
            
            if (predictedClass == classIndex && trueClass == classIndex) {
                truePositives++;
            } else if (predictedClass == classIndex && trueClass != classIndex) {
                falsePositives++;
            }
        }
        
        if (truePositives + falsePositives == 0) {
            return 0.0;
        }
        
        return (double) truePositives / (truePositives + falsePositives);
    }
    
    /**
     * Compute recall for a specific class
     * 
     * Recall = TP / (TP + FN)
     * 
     * @param inputs Array of input samples
     * @param labels Array of true labels (class indices)
     * @param classIndex The class index to compute recall for
     * @return Recall (0.0 to 1.0)
     */
    public double computeRecall(double[][] inputs, int[][] labels, int classIndex) {
        if (inputs.length != labels.length) {
            throw new IllegalArgumentException("Number of inputs and labels must match");
        }
        
        int truePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            if (labels[i] == null || labels[i].length == 0) {
                throw new IllegalArgumentException("Label at index " + i + " is null or empty");
            }
            
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
            if (trueClass < 0 || trueClass >= outputSize) {
                throw new IllegalArgumentException("Label at index " + i + " is out of bounds: " + trueClass + " (expected 0-" + (outputSize - 1) + ")");
            }
            
            if (predictedClass == classIndex && trueClass == classIndex) {
                truePositives++;
            } else if (predictedClass != classIndex && trueClass == classIndex) {
                falseNegatives++;
            }
        }
        
        if (truePositives + falseNegatives == 0) {
            return 0.0;
        }
        
        return (double) truePositives / (truePositives + falseNegatives);
    }
    
    /**
     * Compute F1 score for a specific class
     * 
     * F1 = 2 * (Precision * Recall) / (Precision + Recall)
     * 
     * @param inputs Array of input samples
     * @param labels Array of true labels (class indices)
     * @param classIndex The class index to compute F1 score for
     * @return F1 score (0.0 to 1.0)
     */
    public double computeF1Score(double[][] inputs, int[][] labels, int classIndex) {
        double precision = computePrecision(inputs, labels, classIndex);
        double recall = computeRecall(inputs, labels, classIndex);
        
        if (precision + recall == 0) {
            return 0.0;
        }
        
        return 2 * (precision * recall) / (precision + recall);
    }
    
    /**
     * Compute confusion matrix
     * 
     * confusionMatrix[i][j] = number of samples with true class i predicted as class j
     * 
     * @param inputs Array of input samples
     * @param labels Array of true labels (class indices)
     * @return 2D confusion matrix
     */
    public int[][] computeConfusionMatrix(double[][] inputs, int[][] labels) {
        if (inputs.length != labels.length) {
            throw new IllegalArgumentException("Number of inputs and labels must match");
        }
        
        int[][] confusionMatrix = new int[outputSize][outputSize];
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
            if (trueClass >= 0 && trueClass < outputSize && 
                predictedClass >= 0 && predictedClass < outputSize) {
                confusionMatrix[trueClass][predictedClass]++;
            }
        }
        
        return confusionMatrix;
    }
    
    private int argMax(double[] array) {
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Array cannot be null or empty");
        }
        
        int maxIndex = 0;
        double maxValue = array[0];
        
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * Trains the network for a specified number of epochs.
     * 
     * <p>This is the main training method that handles epoch management, learning
     * rate scheduling, early stopping, and training history collection. Training
     * data is shuffled at the beginning of each epoch for better generalization.</p>
     * 
     * <p>Training progress is logged to stdout. To disable logging, redirect stdout
     * or modify the implementation.</p>
     * 
     * @param inputs  Training input samples with shape [numSamples, inputSize]
     * @param targets Training target values with shape [numSamples, outputSize]
     * @param epochs  Number of training epochs (must be positive)
     * @return Array of loss values, one per epoch (type depends on configured loss function)
     * @throws IllegalArgumentException if inputs/targets dimensions don't match
     *         or if epochs is not positive
     */
    public double[] trainEpochs(double[][] inputs, double[][] targets, int epochs) {
        return trainEpochs(inputs, targets, epochs, lossType);
    }
    
    /**
     * Trains the network for a specified number of epochs with a specific loss function.
     * 
     * @param inputs   Training input samples
     * @param targets  Training target values
     * @param epochs   Number of training epochs
     * @param lossType The loss function to use for training
     * @return Training history (loss per epoch)
     */
    public double[] trainEpochs(double[][] inputs, double[][] targets, int epochs, LossType lossType) {
        double[] history = new double[epochs];
        
        if (earlyStoppingEnabled) {
            resetEarlyStopping();
        }
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] indices = new int[inputs.length];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            
            for (int i = indices.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            double[][] shuffledInputs = new double[inputs.length][];
            double[][] shuffledTargets = new double[targets.length][];
            
            for (int i = 0; i < indices.length; i++) {
                shuffledInputs[i] = inputs[indices[i]];
                shuffledTargets[i] = targets[indices[i]];
            }
            
            train(shuffledInputs, shuffledTargets, lossType);
            
            // Apply learning rate schedule
            if (lrSchedule != LearningRateSchedule.NONE) {
                learningRate = computeLearningRate(epoch, epochs);
            }
            
            if (lossType == LossType.MSE) {
                history[epoch] = computeMSE(inputs, targets);
            } else if (lossType == LossType.MAE) {
                history[epoch] = computeMAE(inputs, targets);
            } else if (lossType == LossType.HUBER) {
                history[epoch] = computeHuber(inputs, targets);
            } else {
                history[epoch] = computeCrossEntropy(inputs, targets);
            }
            
            // Check early stopping
            if (earlyStoppingEnabled) {
                double currentLoss = history[epoch];
                
                if (currentLoss < bestValidationLoss - earlyStoppingMinDelta) {
                    bestValidationLoss = currentLoss;
                    epochsWithoutImprovement = 0;
                    saveBestWeights();
                } else {
                    epochsWithoutImprovement++;
                }
                
                if (epochsWithoutImprovement >= earlyStoppingPatience) {
                    earlyStoppingTriggered = true;
                    restoreBestWeights();
                    System.out.println("Early stopping triggered at epoch " + (epoch + 1) + 
                                     " (best loss: " + String.format("%.6f", bestValidationLoss) + ")");
                    // Fill remaining history with best loss
                    for (int e = epoch + 1; e < epochs; e++) {
                        history[e] = bestValidationLoss;
                    }
                    break;
                }
            }
        }
        
        return history;
    }
    
    /**
     * Validate file path to prevent path traversal attacks
     * 
     * @param filePath The file path to validate
     * @throws IllegalArgumentException If path contains dangerous sequences
     */
    private static void validateFilePath(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        
        String normalizedPath = filePath.replace('\\', '/');
        
        if (normalizedPath.contains("../") || normalizedPath.contains("..\\")) {
            throw new IllegalArgumentException("Path traversal not allowed: " + filePath);
        }
        
        if (normalizedPath.startsWith("/") || normalizedPath.matches("^[a-zA-Z]:/.*")) {
            throw new IllegalArgumentException("Absolute paths not allowed: " + filePath);
        }
    }
    
    /**
     * Validate that a value is a valid number (not NaN or Infinity)
     * 
     * @param value The value to check
     * @param fieldName The name of the field for error messages
     */
    private static void validateNumericValue(double value, String fieldName) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException("Invalid " + fieldName + ": " + value);
        }
    }
    
    /**
     * Saves the trained model to a file.
     * 
     * <p>The saved file includes:</p>
     * <ul>
     *   <li>Network architecture (layer sizes, activations)</li>
     *   <li>Hyperparameters (learning rate, momentum, regularization)</li>
     *   <li>All weights and biases</li>
     *   <li>Loss function type</li>
     * </ul>
     * 
     * <p>Warning: The file format may change between versions. Loading a model
     * saved with a different version may fail.</p>
     * 
     * @param filePath Path to save the model (will be overwritten if exists)
     * @throws IOException if file cannot be written
     * @throws SecurityException if write permission is denied
     */
    public void saveModel(String filePath) throws IOException {
        validateFilePath(filePath);
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println(inputSize);
            
            writer.println(hiddenLayers.length);
            for (int size : hiddenLayers) {
                writer.println(size);
            }
            
            writer.println(outputSize);
            
            writer.println(activations.length);
            for (String activation : activations) {
                writer.println(activation);
            }
            
            writer.println(learningRate);
            writer.println(momentum);
            writer.println(regularization);
            writer.println(batchSize);
            
            writer.println(lossType.name());
            
            for (int l = 0; l < weights.length; l++) {
                writer.println(weights[l].length);
                writer.println(weights[l][0].length);
                
                for (int n = 0; n < weights[l].length; n++) {
                    for (int i = 0; i < weights[l][n].length; i++) {
                        writer.println(weights[l][n][i]);
                    }
                }
            }
            
            for (int l = 0; l < biases.length; l++) {
                writer.println(biases[l].length);
                for (int n = 0; n < biases[l].length; n++) {
                    writer.println(biases[l][n]);
                }
            }
        }
    }
    
    /**
     * Loads a trained model from a file.
     * 
     * <p>Creates a new FlexNet instance with weights and configuration restored
     * from the saved file. The loaded model can be used immediately for inference.</p>
     * 
     * @param filePath Path to the saved model file
     * @return Loaded FlexNet instance with restored weights and configuration
     * @throws IOException if file cannot be read or format is invalid
     * @throws NumberFormatException if file contains invalid data
     */
    public static FlexNet loadModel(String filePath) throws IOException {
        validateFilePath(filePath);
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            int inputSize = Integer.parseInt(reader.readLine().trim());
            
            if (inputSize <= 0) {
                throw new IllegalArgumentException("Invalid inputSize: " + inputSize);
            }
            
            int numHidden = Integer.parseInt(reader.readLine().trim());
            if (numHidden < 0) {
                throw new IllegalArgumentException("Invalid numHidden: " + numHidden);
            }
            
            int[] hiddenLayers = new int[numHidden];
            for (int i = 0; i < numHidden; i++) {
                hiddenLayers[i] = Integer.parseInt(reader.readLine().trim());
                if (hiddenLayers[i] <= 0) {
                    throw new IllegalArgumentException("Invalid hidden layer size at index " + i + ": " + hiddenLayers[i]);
                }
            }
            
            int outputSize = Integer.parseInt(reader.readLine().trim());
            if (outputSize <= 0) {
                throw new IllegalArgumentException("Invalid outputSize: " + outputSize);
            }
            
            int numActivations = Integer.parseInt(reader.readLine().trim());
            if (numActivations <= 0 || numActivations > 100) {
                throw new IllegalArgumentException("Invalid numActivations: " + numActivations);
            }
            
            String[] activations = new String[numActivations];
            for (int i = 0; i < numActivations; i++) {
                activations[i] = reader.readLine().trim();
            }
            
            double learningRate = Double.parseDouble(reader.readLine().trim());
            validateNumericValue(learningRate, "learningRate");
            
            double momentum = Double.parseDouble(reader.readLine().trim());
            validateNumericValue(momentum, "momentum");
            
            double regularization = Double.parseDouble(reader.readLine().trim());
            validateNumericValue(regularization, "regularization");
            
            int batchSize = Integer.parseInt(reader.readLine().trim());
            if (batchSize <= 0) {
                throw new IllegalArgumentException("Invalid batchSize: " + batchSize);
            }
            
            LossType lossType = LossType.valueOf(reader.readLine().trim());
            
            FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                          activations, learningRate, momentum,
                                          regularization, batchSize);
            network.setLossType(lossType);
            
            for (int l = 0; l < network.weights.length; l++) {
                int numNeurons = Integer.parseInt(reader.readLine().trim());
                int numInputs = Integer.parseInt(reader.readLine().trim());
                
                if (numNeurons <= 0 || numNeurons > 100000 || numInputs <= 0 || numInputs > 100000) {
                    throw new IllegalArgumentException("Invalid layer dimensions at layer " + l + ": neurons=" + numNeurons + ", inputs=" + numInputs);
                }
                
                for (int n = 0; n < numNeurons; n++) {
                    for (int i = 0; i < numInputs; i++) {
                        double weight = Double.parseDouble(reader.readLine().trim());
                        validateNumericValue(weight, "weight at layer " + l + ", neuron " + n + ", input " + i);
                        network.weights[l][n][i] = weight;
                    }
                }
            }
            
            for (int l = 0; l < network.biases.length; l++) {
                int numBiases = Integer.parseInt(reader.readLine().trim());
                for (int n = 0; n < numBiases; n++) {
                    double bias = Double.parseDouble(reader.readLine().trim());
                    validateNumericValue(bias, "bias at layer " + l + ", neuron " + n);
                    network.biases[l][n] = bias;
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
     * @param learningRate New learning rate
     */
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new InvalidArchitectureException(
                "Learning rate must be positive",
                "learningRate=" + learningRate);
        }
        this.learningRate = learningRate;
    }
    
    /**
     * Get the gradient clipping threshold
     * 
     * @return Gradient clip threshold (0 means no clipping)
     */
    public double getGradientClipThreshold() {
        return gradientClipThreshold;
    }
    
    /**
     * Set gradient clipping threshold to prevent exploding gradients
     * 
     * @param gradientClipThreshold Threshold for gradient clipping (0 to disable)
     */
    public void setGradientClipThreshold(double gradientClipThreshold) {
        if (gradientClipThreshold < 0) {
            throw new IllegalArgumentException("Gradient clip threshold must be non-negative");
        }
        this.gradientClipThreshold = gradientClipThreshold;
    }
    
    /**
     * Get the Huber loss delta threshold
     * 
     * @return Huber delta (default 1.0)
     */
    public double getHuberDelta() {
        return huberDelta;
    }
    
    /**
     * Set the Huber loss delta threshold
     * 
     * Huber loss is quadratic for |error| <= delta, linear for |error| > delta
     * 
     * @param huberDelta Threshold between quadratic and linear regions
     */
    public void setHuberDelta(double huberDelta) {
        if (huberDelta <= 0) {
            throw new IllegalArgumentException("Huber delta must be positive");
        }
        this.huberDelta = huberDelta;
    }
    
    /**
     * Get the current learning rate schedule
     * 
     * @return The learning rate schedule type
     */
    public LearningRateSchedule getLearningRateSchedule() {
        return lrSchedule;
    }
    
    /**
     * Set learning rate schedule to step decay
     * 
     * Learning rate is multiplied by gamma every stepSize epochs
     * 
     * @param stepSize Number of epochs between LR reductions
     * @param gamma Factor to multiply LR by (e.g., 0.5 for halving)
     */
    public void setStepLearningRate(int stepSize, double gamma) {
        if (stepSize <= 0) {
            throw new IllegalArgumentException("Step size must be positive");
        }
        if (gamma <= 0 || gamma > 1) {
            throw new IllegalArgumentException("Gamma must be between 0 and 1");
        }
        this.lrSchedule = LearningRateSchedule.STEP;
        this.stepSize = stepSize;
        this.stepGamma = gamma;
    }
    
    /**
     * Set learning rate to exponential decay
     * 
     * LR = initialLR * (decayRate ^ epoch)
     * 
     * @param decayRate The decay rate (e.g., 0.95)
     */
    public void setExponentialLearningRate(double decayRate) {
        if (decayRate <= 0 || decayRate > 1) {
            throw new IllegalArgumentException("Decay rate must be between 0 and 1");
        }
        this.lrSchedule = LearningRateSchedule.EXPONENTIAL;
        this.exponentialDecay = decayRate;
    }
    
    /**
     * Set learning rate to cosine annealing
     * 
     * LR decays from initialLR to 0 following half of cosine curve
     * 
     * @param period Number of epochs for one full cycle
     */
    public void setCosineLearningRate(int period) {
        if (period <= 0) {
            throw new IllegalArgumentException("Period must be positive");
        }
        this.lrSchedule = LearningRateSchedule.COSINE;
        this.cosinePeriod = period;
    }
    
    /**
     * Disable learning rate scheduling (use constant LR)
     */
    public void disableLearningRateSchedule() {
        this.lrSchedule = LearningRateSchedule.NONE;
        this.learningRate = initialLearningRate;
    }
    
    /**
     * Compute learning rate based on current epoch and schedule
     * 
     * @param epoch Current epoch number
     * @param totalEpochs Total number of epochs
     * @return The computed learning rate
     */
    private double computeLearningRate(int epoch, int totalEpochs) {
        switch (lrSchedule) {
            case STEP:
                int decays = epoch / stepSize;
                return initialLearningRate * Math.pow(stepGamma, decays);
                
            case EXPONENTIAL:
                return initialLearningRate * Math.pow(exponentialDecay, epoch);
                
            case COSINE:
                double progress = Math.PI * epoch / cosinePeriod;
                if (progress > Math.PI) progress = Math.PI;
                return initialLearningRate * 0.5 * (1.0 + Math.cos(progress));
                
            default:
                return initialLearningRate;
        }
    }
    
    /**
     * Enable Adam optimizer
     * 
     * Adam (Adaptive Moment Estimation) combines momentum with RMSProp
     * - Uses first moment (momentum-like) and second moment (RMSProp-like)
     * - Bias correction for accurate estimates early in training
     * 
     * @param learningRate Learning rate for Adam
     */
    public void enableAdam(double learningRate) {
        this.useAdam = true;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        
        if (weights != null) {
            initializeAdam();
        }
    }
    
    /**
     * Enable Adam with custom hyperparameters
     * 
     * @param learningRate Learning rate
     * @param beta1 First moment decay (typically 0.9)
     * @param beta2 Second moment decay (typically 0.999)
     * @param epsilon Small constant for numerical stability (typically 1e-8)
     */
    public void enableAdam(double learningRate, double beta1, double beta2, double epsilon) {
        if (beta1 <= 0 || beta1 >= 1) {
            throw new IllegalArgumentException("Beta1 must be between 0 and 1");
        }
        if (beta2 <= 0 || beta2 >= 1) {
            throw new IllegalArgumentException("Beta2 must be between 0 and 1");
        }
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }
        
        this.useAdam = true;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.adamBeta1 = beta1;
        this.adamBeta2 = beta2;
        this.adamEpsilon = epsilon;
        
        if (weights != null) {
            initializeAdam();
        }
    }
    
    /**
     * Disable Adam optimizer and use standard momentum
     */
    public void disableAdam() {
        this.useAdam = false;
    }
    
    /**
     * Check if Adam optimizer is enabled
     * 
     * @return True if using Adam
     */
    public boolean isUsingAdam() {
        return useAdam;
    }
    
    /**
     * Enable early stopping to prevent overfitting
     * 
     * Training stops when validation loss does not improve by minDelta for patience epochs
     * 
     * @param patience Number of epochs to wait for improvement
     * @param minDelta Minimum improvement to count as improvement
     */
    public void enableEarlyStopping(int patience, double minDelta) {
        if (patience <= 0) {
            throw new IllegalArgumentException("Patience must be positive");
        }
        if (minDelta < 0) {
            throw new IllegalArgumentException("Min delta must be non-negative");
        }
        this.earlyStoppingEnabled = true;
        this.earlyStoppingPatience = patience;
        this.earlyStoppingMinDelta = minDelta;
        this.bestValidationLoss = Double.MAX_VALUE;
        this.epochsWithoutImprovement = 0;
        this.earlyStoppingTriggered = false;
    }
    
    /**
     * Disable early stopping
     */
    public void disableEarlyStopping() {
        this.earlyStoppingEnabled = false;
    }
    
    /**
     * Check if early stopping was triggered
     * 
     * @return True if early stopping stopped training
     */
    public boolean wasEarlyStoppingTriggered() {
        return earlyStoppingTriggered;
    }
    
    /**
     * Get number of epochs without improvement
     * 
     * @return Epochs since best validation loss
     */
    public int getEpochsWithoutImprovement() {
        return epochsWithoutImprovement;
    }
    
    /**
     * Reset early stopping state for new training
     */
    private void resetEarlyStopping() {
        this.bestValidationLoss = Double.MAX_VALUE;
        this.epochsWithoutImprovement = 0;
        this.earlyStoppingTriggered = false;
        this.bestWeights = null;
        this.bestBiases = null;
    }
    
    /**
     * Save current weights as best
     */
    private void saveBestWeights() {
        this.bestWeights = new double[weights.length][][];
        this.bestBiases = new double[biases.length][];
        
        for (int l = 0; l < weights.length; l++) {
            bestWeights[l] = new double[weights[l].length][];
            for (int n = 0; n < weights[l].length; n++) {
                bestWeights[l][n] = weights[l][n].clone();
            }
            bestBiases[l] = biases[l].clone();
        }
    }
    
    /**
     * Restore best weights
     */
    private void restoreBestWeights() {
        if (bestWeights == null) return;
        
        for (int l = 0; l < weights.length; l++) {
            for (int n = 0; n < weights[l].length; n++) {
                System.arraycopy(bestWeights[l][n], 0, weights[l][n], 0, bestWeights[l][n].length);
            }
            System.arraycopy(bestBiases[l], 0, biases[l], 0, bestBiases[l].length);
        }
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
     * @param momentum New momentum coefficient
     */
    public void setMomentum(double momentum) {
        if (momentum < 0 || momentum >= 1) {
            throw new InvalidArchitectureException(
                "Momentum must be between 0 and 1",
                "momentum=" + momentum);
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
     * L2 regularization adds a penalty term: L_total = L_original + λ * Σ(w²)
     * 
     * @param regularization New regularization strength
     */
    public void setRegularization(double regularization) {
        if (regularization < 0) {
            throw new InvalidArchitectureException(
                "Regularization must be non-negative",
                "regularization=" + regularization);
        }
        if (regularization > 1.0) {
            throw new IllegalArgumentException("Regularization must be <= 1.0 to prevent numerical instability");
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
     * @param batchSize New batch size
     */
    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new InvalidArchitectureException(
                "Batch size must be positive",
                "batchSize=" + batchSize);
        }
        this.batchSize = batchSize;
    }
    
    /**
     * Get the loss type
     * 
     * @return Current loss type
     */
    public LossType getLossType() {
        return lossType;
    }
    
    /**
     * Set the loss type
     * 
     * @param lossType New loss type
     */
    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }
    
    /**
     * Enables dropout regularization.
     * 
     * <p>Dropout is a regularization technique that randomly sets a fraction of
     * neurons to 0 during each training iteration, preventing overfitting.
     * Uses inverted dropout: activations are scaled by 1/(1-dropoutRate) during
     * training so no scaling is needed at test time.</p>
     * 
     * <p>Typical dropout rates: 0.1 to 0.5</p>
     * 
     * @param dropoutRate Probability of dropping a neuron (0.0 to 1.0)
     * @throws InvalidArchitectureException if dropoutRate is not in range [0, 1)
     */
    public void enableDropout(double dropoutRate) {
        if (dropoutRate < 0 || dropoutRate >= 1) {
            throw new InvalidArchitectureException(
                "Dropout rate must be in range [0, 1)",
                "dropoutRate=" + dropoutRate);
        }
        this.dropoutRate = dropoutRate;
        this.dropoutEnabled = true;
    }
    
    /**
     * Disables dropout regularization.
     */
    public void disableDropout() {
        this.dropoutEnabled = false;
    }
    
    /**
     * Gets the current dropout rate.
     * 
     * @return The dropout probability (0.0 if disabled)
     */
    public double getDropoutRate() {
        return dropoutRate;
    }
    
    /**
     * Checks if dropout is enabled.
     * 
     * @return true if dropout is enabled
     */
    public boolean isDropoutEnabled() {
        return dropoutEnabled;
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
            total += weights[l].length * weights[l][0].length;
            total += biases[l].length;
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
        System.out.println("Loss Type: " + lossType);
        System.out.println("Total Parameters: " + getTotalParameters());
    }
    
    /**
     * Main method for testing and demonstration
     */
    public static void main(String[] args) {
        System.out.println("FlexNet - Neural Network Demonstration");
        System.out.println("========================================\n");
        
        int inputSize = 2;
        int[] hiddenLayers = {4};
        int outputSize = 1;
        String[] activations = {"tanh", "sigmoid"};
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                      activations, 0.5, 0.9, 0.001, 32);
        
        network.printArchitecture();
        System.out.println();
        
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
        
        double[] losses = network.trainEpochs(inputs, targets, 10000);
        
        System.out.println("Final MSE: " + losses[losses.length - 1]);
        System.out.println("\nPredictions after training:");
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            System.out.printf("Input: [%d, %d] -> Output: %.4f (Target: %.0f)%n",
                            (int)inputs[i][0], (int)inputs[i][1],
                            prediction[0], targets[i][0]);
        }
        
        try {
            network.saveModel("FlexNet_xor_model.txt");
            System.out.println("\nModel saved to FlexNet_xor_model.txt");
            
            FlexNet loadedNetwork = FlexNet.loadModel("FlexNet_xor_model.txt");
            System.out.println("Model loaded successfully");
            
            double[] pred1 = network.predict(inputs[0]);
            double[] pred2 = loadedNetwork.predict(inputs[0]);
            System.out.printf("Verification: Original: %.4f, Loaded: %.4f%n", 
                            pred1[0], pred2[0]);
            
        } catch (IOException e) {
            System.err.println("Error saving/loading model: " + e.getMessage());
        }
        
        System.out.println("\n========================================");
        System.out.println("Regression Demonstration: f(x) = sin(x)");
        System.out.println("========================================");
        
        int numSamples = 100;
        double[][] regInputs = new double[numSamples][1];
        double[][] regTargets = new double[numSamples][1];
        
        for (int i = 0; i < numSamples; i++) {
            double x = -Math.PI + (2 * Math.PI * i) / (numSamples - 1);
            regInputs[i][0] = x;
            regTargets[i][0] = Math.sin(x);
        }
        
        FlexNet regNetwork = new FlexNet(1, new int[]{16, 16}, 1,
                                         new String[]{"tanh", "tanh", "linear"},
                                         0.01, 0.0, 0.0001, 16);
        
        System.out.println("Training regression network...");
        
        regNetwork.trainEpochs(regInputs, regTargets, 500);
        
        double mse = regNetwork.computeMSE(regInputs, regTargets);
        System.out.println("Final MSE: " + mse);
        
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
