import java.io.*;
import java.util.*;

/**
 * FlexNet - A flexible multi-layer perceptron (fully connected neural network) implementation
 * 
 * This class implements a complete neural network with:
 * - Configurable architecture (input, hidden layers, output)
 * - Multiple activation functions (sigmoid, tanh, relu, linear, elu, softmax)
 * - Xavier/Glorot and He weight initialization
 * - Backpropagation with momentum and L2 regularization
 * - Batch, mini-batch, and stochastic gradient descent
 * - Model serialization/deserialization
 * - Multiple loss functions (MSE, Cross-Entropy)
 * - Training metrics (accuracy, precision, recall, F1, confusion matrix)
 * 
 * @author FlexNet Implementation
 * @version 1.1
 */

enum LossType {
    MSE,
    CROSS_ENTROPY,
    MAE
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
    
    private Random random;
    
    private double[][] preActivations;
    private double[][] postActivations;
    
    private double[][] deltas;
    
    private LossType lossType;
    
    private static final double ELU_ALPHA = 1.0;
    
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
        this.gradientClipThreshold = 0.0;
        this.random = new Random();
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
     * Feedforward computation - computes the network output for given input
     * 
     * For each layer l:
     *   z_l = W_l * a_{l-1} + b_l    (pre-activation)
     *   a_l = activation(z_l)       (post-activation)
     * 
     * @param input Input features (size inputSize)
     * @return Output predictions (size outputSize)
     */
    public double[] predict(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch. Expected " + inputSize + 
                                               ", got " + input.length);
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
                
                for (int i = 0; i < values.length; i++) {
                    result[i] /= sum;
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
            }
        }
        
        double[] prevActivations = input;
        
        for (int l = 0; l < numLayers; l++) {
            if (l > 0) {
                prevActivations = postActivations[l - 1];
            }
            
            for (int n = 0; n < weights[l].length; n++) {
                double delta = deltas[l][n];
                
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
            
            for (int l = 0; l < numLayers; l++) {
                for (int n = 0; n < weights[l].length; n++) {
                    double avgGradB = gradB[l][n] / batchActualSize;
                    
                    if (gradientClipThreshold > 0) {
                        avgGradB = clipGradient(avgGradB);
                    }
                    
                    velocityB[l][n] = momentum * velocityB[l][n] - 
                                     learningRate * avgGradB;
                    biases[l][n] += velocityB[l][n];
                    
                    for (int i2 = 0; i2 < weights[l][n].length; i2++) {
                        double avgGradW = gradW[l][n][i2] / batchActualSize +
                                         regularization * weights[l][n][i2];
                        
                        if (gradientClipThreshold > 0) {
                            avgGradW = clipGradient(avgGradW);
                        }
                        
                        velocityW[l][n][i2] = momentum * velocityW[l][n][i2] - 
                                             learningRate * avgGradW;
                        weights[l][n][i2] += velocityW[l][n][i2];
                    }
                }
            }
        }
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
     * Compute Mean Squared Error loss
     * 
     * MSE = (1/n) * Σ(y_true - y_pred)²
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
     * Compute Cross-Entropy loss
     * 
     * L = -sum(y_true * log(y_pred))
     * 
     * @param inputs Array of input samples
     * @param targets Array of target outputs (one-hot encoded or class indices)
     * @return Cross-Entropy loss
     */
    public double computeCrossEntropy(double[][] inputs, double[][] targets) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of inputs and targets must match");
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
     * Compute classification accuracy
     * 
     * @param inputs Array of input samples
     * @param labels Array of true labels (class indices)
     * @return Accuracy (0.0 to 1.0)
     */
    public double computeAccuracy(double[][] inputs, int[][] labels) {
        if (inputs.length != labels.length) {
            throw new IllegalArgumentException("Number of inputs and labels must match");
        }
        
        int correct = 0;
        
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
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
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
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
            double[] prediction = predict(inputs[i]);
            int predictedClass = argMax(prediction);
            int trueClass = labels[i][0];
            
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
     * Train the network for a specified number of epochs
     * 
     * @param inputs Training inputs
     * @param targets Training targets
     * @param epochs Number of training epochs
     * @return Training history (MSE per epoch)
     */
    public double[] trainEpochs(double[][] inputs, double[][] targets, int epochs) {
        return trainEpochs(inputs, targets, epochs, lossType);
    }
    
    /**
     * Train the network for a specified number of epochs with specified loss type
     * 
     * @param inputs Training inputs
     * @param targets Training targets
     * @param epochs Number of training epochs
     * @param lossType The loss function to use
     * @return Training history (loss per epoch)
     */
    public double[] trainEpochs(double[][] inputs, double[][] targets, int epochs, LossType lossType) {
        double[] history = new double[epochs];
        
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
            
            if (lossType == LossType.MSE) {
                history[epoch] = computeMSE(inputs, targets);
            } else if (lossType == LossType.MAE) {
                history[epoch] = computeMAE(inputs, targets);
            } else {
                history[epoch] = computeCrossEntropy(inputs, targets);
            }
        }
        
        return history;
    }
    
    /**
     * Save the model to a file
     * 
     * @param filePath Path to save the model
     * @throws IOException If file cannot be written
     */
    public void saveModel(String filePath) throws IOException {
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
     * Load a model from a file
     * 
     * @param filePath Path to the model file
     * @return Loaded FlexNet model
     * @throws IOException If file cannot be read
     */
    public static FlexNet loadModel(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            int inputSize = Integer.parseInt(reader.readLine().trim());
            
            int numHidden = Integer.parseInt(reader.readLine().trim());
            int[] hiddenLayers = new int[numHidden];
            for (int i = 0; i < numHidden; i++) {
                hiddenLayers[i] = Integer.parseInt(reader.readLine().trim());
            }
            
            int outputSize = Integer.parseInt(reader.readLine().trim());
            
            int numActivations = Integer.parseInt(reader.readLine().trim());
            String[] activations = new String[numActivations];
            for (int i = 0; i < numActivations; i++) {
                activations[i] = reader.readLine().trim();
            }
            
            double learningRate = Double.parseDouble(reader.readLine().trim());
            double momentum = Double.parseDouble(reader.readLine().trim());
            double regularization = Double.parseDouble(reader.readLine().trim());
            int batchSize = Integer.parseInt(reader.readLine().trim());
            
            LossType lossType = LossType.valueOf(reader.readLine().trim());
            
            FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                          activations, learningRate, momentum,
                                          regularization, batchSize);
            network.setLossType(lossType);
            
            for (int l = 0; l < network.weights.length; l++) {
                int numNeurons = Integer.parseInt(reader.readLine().trim());
                int numInputs = Integer.parseInt(reader.readLine().trim());
                
                for (int n = 0; n < numNeurons; n++) {
                    for (int i = 0; i < numInputs; i++) {
                        network.weights[l][n][i] = Double.parseDouble(reader.readLine().trim());
                    }
                }
            }
            
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
     * @param learningRate New learning rate
     */
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
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
     * L2 regularization adds a penalty term: L_total = L_original + λ * Σ(w²)
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
     * @param batchSize New batch size
     */
    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
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
