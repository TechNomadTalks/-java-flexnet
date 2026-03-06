/**
 * Demo class demonstrating the FlexNet neural network on the XOR problem.
 * 
 * This demo showcases:
 * - Network creation and configuration
 * - XOR training data definition
 * - Training with loss progression monitoring
 * - Predictions before and after training
 * - Different activation functions and hidden layer configurations
 * - Model saving and loading
 * 
 * XOR Truth Table:
 * Input: [0,0] -> Output: [0]
 * Input: [0,1] -> Output: [1]
 * Input: [1,0] -> Output: [1]
 * Input: [1,1] -> Output: [0]
 */
public class Demo {
    
    public static void main(String[] args) {
        System.out.println("================================================================================");
        System.out.println("                    FlexNet XOR Training Demonstration");
        System.out.println("================================================================================\n");
        
        // ========================================================================
        // SECTION 1: Define XOR Training Data
        // ========================================================================
        System.out.println("SECTION 1: XOR Training Data");
        System.out.println("------------------------------");
        
        // XOR requires 2 input features and produces 1 output
        double[][] inputs = {
            {0.0, 0.0},  // Input 1: [0, 0] -> Expected output: 0
            {0.0, 1.0},  // Input 2: [0, 1] -> Expected output: 1
            {1.0, 0.0},  // Input 3: [1, 0] -> Expected output: 1
            {1.0, 1.0}   // Input 4: [1, 1] -> Expected output: 0
        };
        
        double[][] targets = {
            {0.0},  // Target for [0, 0]
            {1.0},  // Target for [0, 1]
            {1.0},  // Target for [1, 0]
            {0.0}   // Target for [1, 1]
        };
        
        // Display the XOR truth table
        System.out.println("XOR Truth Table:");
        System.out.println("----------------");
        for (int i = 0; i < inputs.length; i++) {
            System.out.printf("  Input: [%.0f, %.0f] -> Output: [%.0f]%n", 
                inputs[i][0], inputs[i][1], targets[i][0]);
        }
        System.out.println();
        
        // ========================================================================
        // SECTION 2: Create and Train Basic Network
        // ========================================================================
        System.out.println("================================================================================");
        System.out.println("SECTION 2: Basic Network Training (tanh activation, 4 hidden neurons)");
        System.out.println("================================================================================\n");
        
        // Network configuration for XOR
        int inputSize = 2;        // 2 input features
        int[] hiddenLayers = {4}; // 4 neurons in hidden layer
        int outputSize = 1;       // 1 output neuron
        
        // Activation functions: first for hidden layer, second for output
        // - tanh: good for hidden layers, outputs values in [-1, 1]
        // - sigmoid: good for output when target is in [0, 1]
        String[] activations = {"tanh", "sigmoid"};
        
        // Hyperparameters
        double learningRate = 0.5;      // Step size for gradient descent (0.1-1.0 is typical)
        double momentum = 0.9;          // Momentum to accelerate convergence
        double regularization = 0.001; // L2 regularization to prevent overfitting
        int batchSize = 4;              // Process all 4 samples at once
        
        // Create the FlexNet network
        FlexNet network = new FlexNet(
            inputSize, 
            hiddenLayers, 
            outputSize, 
            activations, 
            learningRate, 
            momentum, 
            regularization, 
            batchSize
        );
        
        // Display network architecture
        System.out.println("Network Architecture:");
        network.printArchitecture();
        System.out.println();
        
        // ========================================================================
        // SECTION 3: Show Predictions BEFORE Training
        // ========================================================================
        System.out.println("Predictions BEFORE Training:");
        System.out.println("-----------------------------");
        
        double initialMSE = network.computeMSE(inputs, targets);
        System.out.printf("Initial MSE: %.4f%n%n", initialMSE);
        
        System.out.println("Individual predictions:");
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            System.out.printf("  Input: [%.0f, %.0f] -> Prediction: %.4f (Target: %.0f)%n", 
                inputs[i][0], inputs[i][1], prediction[0], targets[i][0]);
        }
        System.out.println();
        
        // ========================================================================
        // SECTION 4: Train the Network
        // ========================================================================
        System.out.println("Training Network...");
        System.out.println("-------------------");
        
        // Training parameters
        int epochs = 2000;  // Number of complete passes through the training data
        
        // Train the network and record loss progression
        double[] lossHistory = network.trainEpochs(inputs, targets, epochs);
        
        // Display loss progression at intervals
        System.out.println("Loss progression (MSE):");
        int[] displayEpochs = {0, 100, 500, 1000, 1500, 2000};
        for (int epoch : displayEpochs) {
            if (epoch < lossHistory.length) {
                System.out.printf("  Epoch %5d: MSE = %.6f%n", epoch, lossHistory[epoch]);
            }
        }
        System.out.println();
        
        // ========================================================================
        // SECTION 5: Show Predictions AFTER Training
        // ========================================================================
        System.out.println("Predictions AFTER Training:");
        System.out.println("---------------------------");
        
        double finalMSE = network.computeMSE(inputs, targets);
        System.out.printf("Final MSE: %.6f%n%n", finalMSE);
        
        System.out.println("Individual predictions:");
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            double roundedPrediction = Math.round(prediction[0]);
            boolean correct = (roundedPrediction == targets[i][0]);
            System.out.printf("  Input: [%.0f, %.0f] -> Prediction: %.4f (Rounded: %.0f) %s%n", 
                inputs[i][0], inputs[i][1], prediction[0], roundedPrediction,
                correct ? "✓ CORRECT" : "✗ INCORRECT");
        }
        System.out.println();
        
        // ========================================================================
        // SECTION 6: Demonstrate Different Configurations
        // ========================================================================
        System.out.println("================================================================================");
        System.out.println("SECTION 6: Different Configurations");
        System.out.println("================================================================================\n");
        
        // Configuration 1: ReLU activation
        System.out.println("Configuration 1: ReLU activation with 4 hidden neurons");
        System.out.println("-------------------------------------------------------");
        
        String[] reluActivations = {"relu", "sigmoid"};
        FlexNet reluNetwork = new FlexNet(2, new int[]{4}, 1, reluActivations, 0.5, 0.9, 0.001, 4);
        
        double[] reluHistory = reluNetwork.trainEpochs(inputs, targets, 2000);
        System.out.printf("Final MSE with ReLU: %.6f%n", reluHistory[reluHistory.length - 1]);
        
        System.out.print("Predictions: ");
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = reluNetwork.predict(inputs[i]);
            System.out.printf("[%.0f,%.0f]->%.0f ", inputs[i][0], inputs[i][1], (double)(double)Math.round(pred[0]));
        }
        System.out.println("\n");
        
        // Configuration 2: More hidden neurons (8)
        System.out.println("Configuration 2: Tanh activation with 8 hidden neurons");
        System.out.println("--------------------------------------------------------");
        
        String[] moreNeuronsActivations = {"tanh", "sigmoid"};
        FlexNet moreNeuronsNetwork = new FlexNet(2, new int[]{8}, 1, moreNeuronsActivations, 0.5, 0.9, 0.001, 4);
        
        double[] moreNeuronsHistory = moreNeuronsNetwork.trainEpochs(inputs, targets, 2000);
        System.out.printf("Final MSE with 8 neurons: %.6f%n", moreNeuronsHistory[moreNeuronsHistory.length - 1]);
        
        System.out.print("Predictions: ");
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = moreNeuronsNetwork.predict(inputs[i]);
            System.out.printf("[%.0f,%.0f]->%.0f ", inputs[i][0], inputs[i][1], (double)Math.round(pred[0]));
        }
        System.out.println("\n");
        
        // Configuration 3: Multiple hidden layers (4, 4)
        System.out.println("Configuration 3: Two hidden layers (4, 4) with tanh activation");
        System.out.println("-----------------------------------------------------------------");
        
        String[] deepActivations = {"tanh", "tanh", "sigmoid"};
        FlexNet deepNetwork = new FlexNet(2, new int[]{4, 4}, 1, deepActivations, 0.5, 0.9, 0.001, 4);
        
        double[] deepHistory = deepNetwork.trainEpochs(inputs, targets, 2000);
        System.out.printf("Final MSE with deep network: %.6f%n", deepHistory[deepHistory.length - 1]);
        
        System.out.print("Predictions: ");
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = deepNetwork.predict(inputs[i]);
            System.out.printf("[%.0f,%.0f]->%.0f ", inputs[i][0], inputs[i][1], (double)Math.round(pred[0]));
        }
        System.out.println("\n");
        
        // Configuration 4: Different learning rate
        System.out.println("Configuration 4: Lower learning rate (0.1) with 4 hidden neurons");
        System.out.println("------------------------------------------------------------------");
        
        String[] slowLearningActivations = {"tanh", "sigmoid"};
        FlexNet slowNetwork = new FlexNet(2, new int[]{4}, 1, slowLearningActivations, 0.1, 0.9, 0.001, 4);
        
        double[] slowHistory = slowNetwork.trainEpochs(inputs, targets, 5000);
        System.out.printf("Final MSE with lower learning rate: %.6f%n", slowHistory[slowHistory.length - 1]);
        
        System.out.print("Predictions: ");
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = slowNetwork.predict(inputs[i]);
            System.out.printf("[%.0f,%.0f]->%.0f ", inputs[i][0], inputs[i][1], (double)Math.round(pred[0]));
        }
        System.out.println("\n");
        
        // ========================================================================
        // SECTION 7: Demonstrate Save and Load Model
        // ========================================================================
        System.out.println("================================================================================");
        System.out.println("SECTION 7: Save and Load Model");
        System.out.println("================================================================================\n");
        
        String modelFilePath = "xor_model.txt";
        
        try {
            // Save the trained network to a file
            System.out.println("Saving trained model to file: " + modelFilePath);
            network.saveModel(modelFilePath);
            System.out.println("Model saved successfully!\n");
            
            // Load the network from the file
            System.out.println("Loading model from file: " + modelFilePath);
            FlexNet loadedNetwork = FlexNet.loadModel(modelFilePath);
            System.out.println("Model loaded successfully!\n");
            
            // Verify the loaded network produces the same results
            System.out.println("Verification - Comparing original and loaded network predictions:");
            System.out.println("--------------------------------------------------------------");
            
            for (int i = 0; i < inputs.length; i++) {
                double[] originalPrediction = network.predict(inputs[i]);
                double[] loadedPrediction = loadedNetwork.predict(inputs[i]);
                
                System.out.printf("Input: [%.0f, %.0f] -> Original: %.4f, Loaded: %.4f, Diff: %.6f%n", 
                    inputs[i][0], inputs[i][1], 
                    originalPrediction[0], 
                    loadedPrediction[0], 
                    Math.abs(originalPrediction[0] - loadedPrediction[0]));
            }
            
            // Compare MSE
            double originalMSE = network.computeMSE(inputs, targets);
            double loadedMSE = loadedNetwork.computeMSE(inputs, targets);
            
            System.out.printf("%nOriginal network MSE: %.6f%n", originalMSE);
            System.out.printf("Loaded network MSE:    %.6f%n", loadedMSE);
            System.out.printf("MSE difference:       %.6f%n", Math.abs(originalMSE - loadedMSE));
            
            System.out.println("\nNote: The loaded network should produce identical results to the original.");
            
        } catch (Exception e) {
            System.err.println("Error during save/load operation: " + e.getMessage());
            e.printStackTrace();
        }
        
        // ========================================================================
        // SECTION 8: Summary
        // ========================================================================
        System.out.println("\n================================================================================");
        System.out.println("SUMMARY");
        System.out.println("================================================================================\n");
        
        System.out.println("Key observations from this demo:");
        System.out.println("---------------------------------");
        System.out.println("1. Before training, the network produces random predictions");
        System.out.println("2. After training, the network learns to correctly classify XOR patterns");
        System.out.println("3. The MSE decreases significantly during training");
        System.out.println("4. Different activation functions and network depths can affect convergence");
        System.out.println("5. The trained model can be saved to disk and reloaded later");
        System.out.println("6. The loaded model produces identical predictions to the original");
        
        System.out.println("\nHyperparameters used in this demo:");
        System.out.println("-----------------------------------");
        System.out.printf("- Learning Rate: %.1f%n", learningRate);
        System.out.printf("- Momentum: %.1f%n", momentum);
        System.out.printf("- Regularization: %.6f%n", regularization);
        System.out.printf("- Batch Size: %d%n", batchSize);
        System.out.printf("- Epochs: %d%n", epochs);
        
        System.out.println("\n================================================================================");
        System.out.println("                         Demo Completed Successfully!");
        System.out.println("================================================================================");
    }
}
