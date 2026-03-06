import java.io.*;
import java.util.*;

/**
 * RegressionDemo - Demonstrates regression tasks using FlexNet neural network
 * 
 * This demo covers three types of regression:
 * 1. Function approximation: sin(x)
 * 2. Polynomial regression: y = 2x^3 - 3x^2 + x - 1
 * 3. Multivariate regression: y = 3*x1 - 2*x2 + x3
 * 
 * Each example shows:
 * - Data generation
 * - Network training with MSE loss
 * - Loss before and after training
 * - Sample predictions vs actual values
 * - Visual-style output showing model fit
 * 
 * @author FlexNet Demo
 * @version 1.0
 */
public class RegressionDemo {
    
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("           FlexNet Regression Demo");
        System.out.println("=".repeat(70));
        System.out.println();
        
        // Run all regression demos
        demoSinFunction();
        System.out.println();
        
        demoPolynomialRegression();
        System.out.println();
        
        demoMultivariateRegression();
        System.out.println();
        
        System.out.println("=".repeat(70));
        System.out.println("           All Demos Complete!");
        System.out.println("=".repeat(70));
    }
    
    /**
     * Demo 1: Function Approximation - sin(x)
     * 
     * Trains a network to approximate y = sin(x) for x in [0, 2*PI]
     */
    private static void demoSinFunction() {
        System.out.println("-".repeat(70));
        System.out.println("DEMO 1: Function Approximation - sin(x)");
        System.out.println("-".repeat(70));
        System.out.println("Target: y = sin(x) for x in [0, 2*PI]");
        System.out.println();
        
        // Generate training data: 100+ points
        int numPoints = 120;
        double[][] inputs = new double[numPoints][1];
        double[][] targets = new double[numPoints][1];
        
        for (int i = 0; i < numPoints; i++) {
            double x = (2 * Math.PI * i) / (numPoints - 1);
            inputs[i][0] = x;
            targets[i][0] = Math.sin(x);
        }
        
        System.out.println("Training Data: " + numPoints + " points");
        System.out.println("X range: [0, " + String.format("%.4f", 2 * Math.PI) + "]");
        System.out.println("Y range: [" + String.format("%.4f", -1.0) + ", " + String.format("%.4f", 1.0) + "]");
        System.out.println();
        
        // Create network: 1 input, [32, 32] hidden, 1 output
        int inputSize = 1;
        int[] hiddenLayers = {32, 32};
        int outputSize = 1;
        String[] activations = {"tanh", "tanh", "linear"};
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                       activations, 0.01, 0.9, 0.0001, 32);
        
        System.out.println("Network Architecture:");
        System.out.println("  Input:  1 neuron");
        System.out.print("  Hidden: ");
        for (int h : hiddenLayers) System.out.print(h + " ");
        System.out.println("neurons (tanh)");
        System.out.println("  Output: 1 neuron (linear)");
        System.out.println();
        
        // Compute initial MSE
        double initialMSE = network.computeMSE(inputs, targets);
        System.out.println("Loss (MSE) before training: " + String.format("%.6f", initialMSE));
        System.out.println();
        
        // Train the network
        System.out.println("Training for 500 epochs...");
        double[] history = network.trainEpochs(inputs, targets, 500);
        
        // Compute final MSE
        double finalMSE = network.computeMSE(inputs, targets);
        System.out.println("Loss (MSE) after training:  " + String.format("%.6f", finalMSE));
        System.out.println("Improvement: " + String.format("%.2f", (initialMSE - finalMSE) / initialMSE * 100) + "%");
        System.out.println();
        
        // Show sample predictions
        System.out.println("Sample Predictions vs Actual:");
        System.out.println("-".repeat(50));
        System.out.printf("%-12s %-12s %-12s %-12s%n", "X", "Actual sin(x)", "Predicted", "Error");
        System.out.println("-".repeat(50));
        
        int[] testIndices = {0, 20, 40, 60, 80, 100, 119};
        for (int idx : testIndices) {
            double[] pred = network.predict(inputs[idx]);
            double actual = targets[idx][0];
            double predicted = pred[0];
            System.out.printf("%-12.4f %-12.4f %-12.4f %-12.4f%n",
                    inputs[idx][0], actual, predicted, actual - predicted);
        }
        System.out.println();
        
        // Visual-style output
        printVisualization("sin(x) Approximation", inputs, targets, network, 0, 0, 
                          -1.5, 1.5, "X", "Y = sin(x)");
    }
    
    /**
     * Demo 2: Polynomial Regression
     * 
     * Trains a network to approximate y = 2x^3 - 3x^2 + x - 1
     */
    private static void demoPolynomialRegression() {
        System.out.println("-".repeat(70));
        System.out.println("DEMO 2: Polynomial Regression");
        System.out.println("-".repeat(70));
        System.out.println("Target: y = 2x^3 - 3x^2 + x - 1 for x in [-2, 2]");
        System.out.println();
        
        // Generate training data: 200+ points
        int numPoints = 200;
        double[][] inputs = new double[numPoints][1];
        double[][] targets = new double[numPoints][1];
        
        for (int i = 0; i < numPoints; i++) {
            double x = -2.0 + (4.0 * i) / (numPoints - 1);
            inputs[i][0] = x;
            targets[i][0] = 2*x*x*x - 3*x*x + x - 1;
        }
        
        System.out.println("Training Data: " + numPoints + " points");
        System.out.println("X range: [-2.0, 2.0]");
        System.out.println();
        
        // Create network: 1 input, [32, 32] hidden, 1 output
        int inputSize = 1;
        int[] hiddenLayers = {32, 32};
        int outputSize = 1;
        String[] activations = {"tanh", "tanh", "linear"};
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                       activations, 0.01, 0.9, 0.0001, 32);
        
        System.out.println("Network Architecture:");
        System.out.println("  Input:  1 neuron");
        System.out.print("  Hidden: ");
        for (int h : hiddenLayers) System.out.print(h + " ");
        System.out.println("neurons (tanh)");
        System.out.println("  Output: 1 neuron (linear)");
        System.out.println();
        
        // Compute initial MSE
        double initialMSE = network.computeMSE(inputs, targets);
        System.out.println("Loss (MSE) before training: " + String.format("%.6f", initialMSE));
        System.out.println();
        
        // Train the network
        System.out.println("Training for 1000 epochs...");
        double[] history = network.trainEpochs(inputs, targets, 1000);
        
        // Compute final MSE
        double finalMSE = network.computeMSE(inputs, targets);
        System.out.println("Loss (MSE) after training:  " + String.format("%.6f", finalMSE));
        System.out.println("Improvement: " + String.format("%.2f", (initialMSE - finalMSE) / initialMSE * 100) + "%");
        System.out.println();
        
        // Show sample predictions
        System.out.println("Sample Predictions vs Actual:");
        System.out.println("-".repeat(60));
        System.out.printf("%-12s %-15s %-15s %-12s%n", "X", "Actual y", "Predicted", "Error");
        System.out.println("-".repeat(60));
        
        int[] testIndices = {0, 40, 80, 100, 120, 160, 199};
        for (int idx : testIndices) {
            double[] pred = network.predict(inputs[idx]);
            double actual = targets[idx][0];
            double predicted = pred[0];
            System.out.printf("%-12.4f %-15.4f %-15.4f %-12.4f%n",
                    inputs[idx][0], actual, predicted, actual - predicted);
        }
        System.out.println();
        
        // Visual-style output
        printVisualization("Polynomial Fit", inputs, targets, network, 0, 0,
                          -20, 20, "X", "Y = 2x^3 - 3x^2 + x - 1");
    }
    
    /**
     * Demo 3: Multivariate Regression
     * 
     * Trains a network to approximate y = 3*x1 - 2*x2 + x3
     */
    private static void demoMultivariateRegression() {
        System.out.println("-".repeat(70));
        System.out.println("DEMO 3: Multivariate Regression");
        System.out.println("-".repeat(70));
        System.out.println("Target: y = 3*x1 - 2*x2 + x3 (3 inputs, 1 output)");
        System.out.println();
        
        // Generate training data: 200+ random samples
        int numPoints = 200;
        double[][] inputs = new double[numPoints][3];
        double[][] targets = new double[numPoints][1];
        
        for (int i = 0; i < numPoints; i++) {
            // Generate random inputs in range [-1, 1]
            double x1 = random.nextDouble() * 2 - 1;
            double x2 = random.nextDouble() * 2 - 1;
            double x3 = random.nextDouble() * 2 - 1;
            
            inputs[i][0] = x1;
            inputs[i][1] = x2;
            inputs[i][2] = x3;
            
            // Compute target: y = 3*x1 - 2*x2 + x3
            targets[i][0] = 3*x1 - 2*x2 + x3;
        }
        
        System.out.println("Training Data: " + numPoints + " random samples");
        System.out.println("Input ranges: x1, x2, x3 in [-1, 1]");
        System.out.println("Output range: [" + String.format("%.2f", -6.0) + ", " + 
                          String.format("%.2f", 6.0) + "] (theoretical)");
        System.out.println();
        
        // Create network: 3 inputs, [32, 32] hidden, 1 output
        int inputSize = 3;
        int[] hiddenLayers = {32, 32};
        int outputSize = 1;
        String[] activations = {"tanh", "tanh", "linear"};
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize,
                                       activations, 0.01, 0.9, 0.0001, 32);
        
        System.out.println("Network Architecture:");
        System.out.println("  Input:  3 neurons (x1, x2, x3)");
        System.out.print("  Hidden: ");
        for (int h : hiddenLayers) System.out.print(h + " ");
        System.out.println("neurons (tanh)");
        System.out.println("  Output: 1 neuron (linear)");
        System.out.println();
        
        // Compute initial MSE
        double initialMSE = network.computeMSE(inputs, targets);
        System.out.println("Loss (MSE) before training: " + String.format("%.6f", initialMSE));
        System.out.println();
        
        // Train the network
        System.out.println("Training for 1000 epochs...");
        double[] history = network.trainEpochs(inputs, targets, 1000);
        
        // Compute final MSE
        double finalMSE = network.computeMSE(inputs, targets);
        System.out.println("Loss (MSE) after training:  " + String.format("%.6f", finalMSE));
        System.out.println("Improvement: " + String.format("%.2f", (initialMSE - finalMSE) / initialMSE * 100) + "%");
        System.out.println();
        
        // Show sample predictions
        System.out.println("Sample Predictions vs Actual:");
        System.out.println("-".repeat(70));
        System.out.printf("%-10s %-10s %-10s %-12s %-12s %-10s%n", 
                        "x1", "x2", "x3", "Actual", "Predicted", "Error");
        System.out.println("-".repeat(70));
        
        // Test with fixed samples
        double[][] testSamples = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
            {-1.0, 1.0, -1.0},
            {0.5, -0.5, 0.5}
        };
        
        double[][] testTargets = new double[testSamples.length][1];
        for (int i = 0; i < testSamples.length; i++) {
            testTargets[i][0] = 3*testSamples[i][0] - 2*testSamples[i][1] + testSamples[i][2];
        }
        
        for (int i = 0; i < testSamples.length; i++) {
            double[] pred = network.predict(testSamples[i]);
            double actual = testTargets[i][0];
            double predicted = pred[0];
            System.out.printf("%-10.4f %-10.4f %-10.4f %-12.4f %-12.4f %-10.4f%n",
                    testSamples[i][0], testSamples[i][1], testSamples[i][2],
                    actual, predicted, actual - predicted);
        }
        System.out.println();
        
        // Show formula learned
        System.out.println("Analysis:");
        System.out.println("--------");
        System.out.println("Target formula: y = 3*x1 - 2*x2 + 1*x3");
        System.out.println("The network learned to approximate this linear combination.");
        System.out.println("Note: With linear output activation, the network can exactly");
        System.out.println("      learn this linear relationship given sufficient training.");
        System.out.println();
        
        // Test set accuracy
        int numTest = 50;
        double[][] testInputs = new double[numTest][3];
        double[][] testTargetsArr = new double[numTest][1];
        
        for (int i = 0; i < numTest; i++) {
            testInputs[i][0] = random.nextDouble() * 2 - 1;
            testInputs[i][1] = random.nextDouble() * 2 - 1;
            testInputs[i][2] = random.nextDouble() * 2 - 1;
            testTargetsArr[i][0] = 3*testInputs[i][0] - 2*testInputs[i][1] + testInputs[i][2];
        }
        
        double testMSE = network.computeMSE(testInputs, testTargetsArr);
        System.out.println("Test Set MSE (50 random samples): " + String.format("%.6f", testMSE));
    }
    
    /**
     * Print a visualization of the regression fit
     * 
     * For 1D inputs, creates an ASCII chart showing actual vs predicted values
     */
    private static void printVisualization(String title, double[][] inputs, 
                                           double[][] targets, FlexNet network,
                                           int inputIdx, int outputIdx,
                                           double yMin, double yMax,
                                           String xLabel, String yLabel) {
        System.out.println("Visual: " + title);
        System.out.println("=".repeat(60));
        
        // Generate smooth prediction curve
        int numPoints = 60;
        double[][] plotInputs = new double[numPoints][1];
        
        double xMin = inputs[0][inputIdx];
        double xMax = inputs[inputs.length - 1][inputIdx];
        
        for (int i = 0; i < numPoints; i++) {
            plotInputs[i][0] = xMin + (xMax - xMin) * i / (numPoints - 1);
        }
        
        // Create a simple ASCII visualization
        int width = 50;
        int height = 15;
        char[][] plot = new char[height][width];
        
        // Initialize with spaces
        for (int i = 0; i < height; i++) {
            Arrays.fill(plot[i], ' ');
        }
        
        // Plot actual data points and predictions
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = network.predict(inputs[i])[outputIdx];
        }
        
        // Map values to grid positions
        for (int i = 0; i < inputs.length; i++) {
            int xPos = (int) ((inputs[i][inputIdx] - xMin) / (xMax - xMin) * (width - 1));
            int yPos = (int) ((yMax - targets[i][outputIdx]) / (yMax - yMin) * (height - 1));
            yPos = Math.max(0, Math.min(height - 1, yPos));
            xPos = Math.max(0, Math.min(width - 1, xPos));
            plot[yPos][xPos] = '*';
        }
        
        // Print the plot
        System.out.println();
        
        // Y-axis label
        System.out.print(yLabel.substring(0, Math.min(10, yLabel.length())));
        System.out.print(" ^ ");
        
        // Print top row
        for (int x = 0; x < width; x++) {
            System.out.print("-");
        }
        System.out.println(">");
        System.out.println(" ".repeat(12) + "|" + " ".repeat(width));
        
        for (int y = 0; y < height; y++) {
            double yVal = yMax - (yMax - yMin) * y / (height - 1);
            System.out.printf("%10.2f |", yVal);
            
            for (int x = 0; x < width; x++) {
                System.out.print(plot[y][x]);
            }
            System.out.println("|");
        }
        
        System.out.println(" ".repeat(12) + "|" + "-".repeat(width));
        System.out.printf("%13s%s%n", " ", xLabel);
        System.out.printf("%13s[%s, %s]%n", " ", 
                         String.format("%.2f", xMin), 
                         String.format("%.2f", xMax));
        
        // Legend
        System.out.println();
        System.out.println("Legend: '*' = data points");
        System.out.println("Note: This shows a scatter plot of training data.");
        System.out.println("      The network learned to fit the underlying function.");
    }
}
