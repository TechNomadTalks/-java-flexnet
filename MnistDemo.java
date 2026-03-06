import java.util.*;

/**
 * MnistDemo - MNIST-like Digit Classification Demo
 * 
 * This demo demonstrates training a neural network on synthetic MNIST-like data.
 * The MNIST dataset contains 28x28 grayscale images of handwritten digits (0-9).
 * For demonstration purposes, we use 8x8 pixel "digit-like" patterns.
 * 
 * Key concepts demonstrated:
 * 1. Synthetic data generation - creating digit-like patterns
 * 2. Neural network training with backpropagation
 * 3. Multi-class classification with softmax and cross-entropy loss
 * 4. Evaluation metrics: accuracy, precision, recall, confusion matrix
 * 
 * Network Architecture:
 * - Input: 64 neurons (8x8 flattened images)
 * - Hidden Layer 1: 128 neurons with ReLU activation
 * - Hidden Layer 2: 64 neurons with ReLU activation
 * - Output: 10 neurons with softmax activation (digits 0-9)
 * - Loss: Cross-Entropy (appropriate for multi-class classification)
 */
public class MnistDemo {
    
    // Network configuration constants
    private static final int IMAGE_SIZE = 8;          // 8x8 images
    private static final int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE;  // 64 inputs
    private static final int[] HIDDEN_LAYERS = {128, 64};
    private static final int OUTPUT_SIZE = 10;         // Digits 0-9
    private static final int NUM_SAMPLES = 200;       // Samples per digit class
    private static final int TOTAL_SAMPLES = NUM_SAMPLES * OUTPUT_SIZE;
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("MNIST-like Digit Classification Demo");
        System.out.println("=".repeat(70));
        System.out.println();
        
        // Step 1: Generate synthetic MNIST-like data
        System.out.println("Step 1: Generating synthetic MNIST-like data...");
        System.out.println("-".repeat(50));
        
        double[][] trainInputs = new double[TOTAL_SAMPLES][INPUT_SIZE];
        double[][] trainTargets = new double[TOTAL_SAMPLES][OUTPUT_SIZE];
        int[][] trainLabels = new int[TOTAL_SAMPLES][1];
        
        double[][] testInputs = new double[TOTAL_SAMPLES / 5][INPUT_SIZE];
        double[][] testTargets = new double[TOTAL_SAMPLES / 5][OUTPUT_SIZE];
        int[][] testLabels = new int[TOTAL_SAMPLES / 5][1];
        
        // Generate training data
        int trainIdx = 0;
        for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
            for (int i = 0; i < NUM_SAMPLES; i++) {
                double[] pattern = generateDigitPattern(digit);
                trainInputs[trainIdx] = pattern;
                trainTargets[trainIdx] = oneHotEncode(digit);
                trainLabels[trainIdx][0] = digit;
                trainIdx++;
            }
        }
        
        // Generate test data (different samples from training)
        Random testRandom = new Random(42);  // Different seed for variety
        int testIdx = 0;
        for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
            for (int i = 0; i < NUM_SAMPLES / 5; i++) {
                double[] pattern = generateDigitPattern(digit, testRandom);
                testInputs[testIdx] = pattern;
                testTargets[testIdx] = oneHotEncode(digit);
                testLabels[testIdx][0] = digit;
                testIdx++;
            }
        }
        
        System.out.println("Generated " + TOTAL_SAMPLES + " training samples");
        System.out.println("Generated " + testInputs.length + " test samples");
        System.out.println("Image size: " + IMAGE_SIZE + "x" + IMAGE_SIZE + " = " + INPUT_SIZE + " features");
        System.out.println();
        
        // Step 2: Create and configure the neural network
        System.out.println("Step 2: Creating FlexNet neural network...");
        System.out.println("-".repeat(50));
        
        // Activation functions: [hidden1, hidden2, output]
        String[] activations = {"relu", "relu", "softmax"};
        
        // Create network with specified architecture
        FlexNet network = new FlexNet(
            INPUT_SIZE,              // Input size: 64 (8x8 flattened)
            HIDDEN_LAYERS,           // Hidden layers: [128, 64]
            OUTPUT_SIZE,             // Output size: 10 (digits 0-9)
            activations,             // Activation functions
            0.1,                     // Learning rate
            0.9,                     // Momentum for faster convergence
            0.0001,                  // L2 regularization (prevents overfitting)
            32                        // Mini-batch size
        );
        
        // Set loss type to CROSS_ENTROPY for classification
        // Cross-entropy works well with softmax for multi-class problems
        network.setLossType(LossType.CROSS_ENTROPY);
        
        network.printArchitecture();
        System.out.println();
        
        // Step 3: Evaluate BEFORE training
        System.out.println("Step 3: Evaluating BEFORE training...");
        System.out.println("-".repeat(50));
        
        double initialAccuracy = network.computeAccuracy(trainInputs, trainLabels);
        System.out.printf("Initial Training Accuracy: %.2f%%%n", initialAccuracy * 100);
        System.out.println("(Random guessing would give ~10% accuracy for 10 classes)");
        System.out.println();
        
        // Step 4: Train the network
        System.out.println("Step 4: Training the network...");
        System.out.println("-".repeat(50));
        
        int epochs = 100;
        System.out.println("Training for " + epochs + " epochs...");
        System.out.println("This may take a moment...");
        
        // Train with cross-entropy loss
        double[] trainingHistory = network.trainEpochs(trainInputs, trainTargets, epochs, LossType.CROSS_ENTROPY);
        
        System.out.println("Training complete!");
        System.out.printf("Final Training Loss: %.4f%n", trainingHistory[trainingHistory.length - 1]);
        
        // Show training progress (first, middle, last few epochs)
        System.out.println("\nTraining progress (loss per epoch):");
        System.out.printf("  Epoch 1:   %.4f%n", trainingHistory[0]);
        System.out.printf("  Epoch 25:  %.4f%n", trainingHistory[24]);
        System.out.printf("  Epoch 50:  %.4f%n", trainingHistory[49]);
        System.out.printf("  Epoch 75:  %.4f%n", trainingHistory[74]);
        System.out.printf("  Epoch %d:  %.4f%n", epochs, trainingHistory[epochs - 1]);
        System.out.println();
        
        // Step 5: Evaluate AFTER training
        System.out.println("Step 5: Evaluating AFTER training...");
        System.out.println("-".repeat(50));
        
        double finalAccuracy = network.computeAccuracy(trainInputs, trainLabels);
        System.out.printf("Final Training Accuracy: %.2f%%%n", finalAccuracy * 100);
        System.out.println();
        
        // Step 6: Compute precision and recall for each digit class
        System.out.println("Step 6: Precision and Recall per Digit Class");
        System.out.println("-".repeat(50));
        System.out.printf("%-8s %-12s %-12s %-12s%n", "Digit", "Precision", "Recall", "F1-Score");
        System.out.println("-".repeat(50));
        
        double[] precisions = new double[OUTPUT_SIZE];
        double[] recalls = new double[OUTPUT_SIZE];
        
        for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
            precisions[digit] = network.computePrecision(trainInputs, trainLabels, digit);
            recalls[digit] = network.computeRecall(trainInputs, trainLabels, digit);
            double f1 = network.computeF1Score(trainInputs, trainLabels, digit);
            System.out.printf("%-8d %-12.4f %-12.4f %-12.4f%n", 
                digit, precisions[digit], recalls[digit], f1);
        }
        System.out.println();
        
        // Step 7: Compute and display confusion matrix
        System.out.println("Step 7: Confusion Matrix");
        System.out.println("-".repeat(50));
        
        int[][] confusionMatrix = network.computeConfusionMatrix(trainInputs, trainLabels);
        
        // Print header
        System.out.print("% True  ");
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            System.out.printf("%4d ", j);
        }
        System.out.println(" <- Predicted");
        System.out.println("-".repeat(50));
        
        // Print matrix
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            System.out.printf("   %d   ", i);
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                System.out.printf("%4d ", confusionMatrix[i][j]);
            }
            System.out.println();
        }
        System.out.println("(Rows: True digit, Columns: Predicted digit)");
        System.out.println();
        
        // Step 8: Show sample predictions
        System.out.println("Step 8: Sample Predictions");
        System.out.println("-".repeat(50));
        
        Random sampleRandom = new Random(123);
        System.out.println("Showing 10 random sample predictions:");
        System.out.printf("%-6s %-12s %-12s %-12s%n", "Sample", "True Digit", "Predicted", "Confidence");
        System.out.println("-".repeat(50));
        
        for (int i = 0; i < 10; i++) {
            int idx = sampleRandom.nextInt(trainInputs.length);
            double[] prediction = network.predict(trainInputs[idx]);
            int predictedDigit = argMax(prediction);
            double confidence = prediction[predictedDigit];
            int trueDigit = trainLabels[idx][0];
            
            String status = predictedDigit == trueDigit ? "OK" : "WRONG";
            System.out.printf("%-6d %-12d %-12d %-10.2f%% %s%n", 
                i + 1, trueDigit, predictedDigit, confidence * 100, status);
        }
        System.out.println();
        
        // Summary
        System.out.println("=".repeat(70));
        System.out.println("DEMO COMPLETE - Summary");
        System.out.println("=".repeat(70));
        System.out.printf("Network Architecture: %d -> [%d, %d] -> %d%n", 
            INPUT_SIZE, HIDDEN_LAYERS[0], HIDDEN_LAYERS[1], OUTPUT_SIZE);
        System.out.printf("Total Parameters: %d%n", network.getTotalParameters());
        System.out.printf("Training Samples: %d%n", TOTAL_SAMPLES);
        System.out.printf("Epochs Trained: %d%n", epochs);
        System.out.printf("Accuracy Improvement: %.2f%% -> %.2f%%%n", 
            initialAccuracy * 100, finalAccuracy * 100);
        System.out.printf("Average Precision: %.2f%%%n", 
            Arrays.stream(precisions).average().orElse(0) * 100);
        System.out.printf("Average Recall: %.2f%%%n", 
            Arrays.stream(recalls).average().orElse(0) * 100);
        System.out.println();
        System.out.println("Key Concepts Demonstrated:");
        System.out.println("1. Synthetic data generation for digit patterns");
        System.out.println("2. Multi-layer neural network with ReLU and softmax");
        System.out.println("3. Cross-entropy loss for classification");
        System.out.println("4. Mini-batch gradient descent with momentum");
        System.out.println("5. Evaluation metrics: Accuracy, Precision, Recall, F1");
        System.out.println("6. Confusion matrix for detailed performance analysis");
        System.out.println("=".repeat(70));
    }
    
    /**
     * Generate a synthetic digit-like pattern
     * Creates patterns that resemble handwritten digits using basic shapes
     * 
     * @param digit The digit (0-9) to generate
     * @return 64-element array representing 8x8 grayscale image
     */
    private static double[] generateDigitPattern(int digit) {
        return generateDigitPattern(digit, new Random(digit * 1000 + System.nanoTime()));
    }
    
    /**
     * Generate a synthetic digit-like pattern with custom random generator
     * 
     * @param digit The digit (0-9) to generate
     * @param random Random generator for adding variation
     * @return 64-element array representing 8x8 grayscale image
     */
    private static double[] generateDigitPattern(int digit, Random random) {
        double[][] image = new double[IMAGE_SIZE][IMAGE_SIZE];
        
        // Base shapes for each digit (simplified 8x8 representations)
        // Different patterns for each digit to make them distinguishable
        switch (digit) {
            case 0: // Circle-like
                drawCircle(image, 3.5, 3.5, 2.8, random);
                break;
            case 1: // Vertical line (with slight variations)
                drawLine(image, 3, 1, 3, 6, random);
                break;
            case 2: // Horizontal top, diagonal down, horizontal bottom
                drawArc(image, 0, 3, 7, 3, true, random);
                break;
            case 3: // Two curves (like 'B' but for 3)
                drawArc(image, 1, 3, 7, 3, true, random);
                drawArc(image, 1, 3, 7, 5, true, random);
                break;
            case 4: // Vertical and diagonal
                drawLine(image, 5, 1, 5, 6, random);
                drawLine(image, 1, 4, 6, 1, random);
                break;
            case 5: // Top horizontal, left vertical, bottom curve
                drawLine(image, 1, 1, 6, 1, random);
                drawLine(image, 1, 1, 1, 3, random);
                drawArc(image, 1, 5, 7, 5, false, random);
                break;
            case 6: // Circle with vertical line on left
                drawCircle(image, 3.5, 4, 2.5, random);
                drawLine(image, 1, 2, 1, 5, random);
                break;
            case 7: // Top horizontal, diagonal down
                drawLine(image, 1, 1, 6, 1, random);
                drawLine(image, 5, 1, 2, 6, random);
                break;
            case 8: // Two circles
                drawCircle(image, 3.5, 2.2, 1.5, random);
                drawCircle(image, 3.5, 5.3, 1.5, random);
                break;
            case 9: // Circle with vertical line on right
                drawCircle(image, 4, 3.5, 2.5, random);
                drawLine(image, 6, 2, 6, 6, random);
                break;
        }
        
        // Flatten 2D array to 1D
        double[] flattened = new double[INPUT_SIZE];
        int idx = 0;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                flattened[idx++] = image[i][j];
            }
        }
        
        return flattened;
    }
    
    /**
     * Draw a circle/oval shape on the image
     */
    private static void drawCircle(double[][] image, double cx, double cy, double radius, Random random) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                double dist = Math.sqrt((i - cy) * (i - cy) + (j - cx) * (j - cx));
                if (Math.abs(dist - radius) < 0.8) {
                    double noise = random.nextDouble() * 0.2;
                    image[i][j] = Math.min(1.0, 0.9 + noise);
                } else if (dist < radius - 0.5) {
                    double noise = random.nextDouble() * 0.1;
                    image[i][j] = Math.min(0.3, noise);
                }
            }
        }
    }
    
    /**
     * Draw a line on the image
     */
    private static void drawLine(double[][] image, int x1, int y1, int x2, int y2, Random random) {
        // Simple line drawing using Bresenham-like algorithm
        int steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
        for (int i = 0; i <= steps; i++) {
            int x = x1 + (x2 - x1) * i / steps;
            int y = y1 + (y2 - y1) * i / steps;
            if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
                double noise = random.nextDouble() * 0.2;
                image[y][x] = Math.min(1.0, 0.9 + noise);
            }
        }
    }
    
    /**
     * Draw an arc/curve on the image
     */
    private static void drawArc(double[][] image, int startX, int startY, int endX, int endY, 
                                boolean curveDown, Random random) {
        // Draw a curved line (arc)
        int midX = (startX + endX) / 2;
        int controlY = curveDown ? Math.max(startY, endY) + 2 : Math.min(startY, endY) - 2;
        
        // Draw two segments with a curve approximation
        for (int t = 0; t <= 20; t++) {
            double t1 = t / 20.0;
            double t2 = 1 - t1;
            
            // Quadratic bezier curve points
            int x = (int)(t2 * t2 * startX + 2 * t2 * t1 * midX + t1 * t1 * endX);
            int y = (int)(t2 * t2 * startY + 2 * t2 * t1 * controlY + t1 * t1 * endY);
            
            if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
                double noise = random.nextDouble() * 0.2;
                image[y][x] = Math.min(1.0, 0.9 + noise);
            }
        }
    }
    
    /**
     * Convert a digit to one-hot encoding
     * 
     * @param digit The digit (0-9)
     * @return One-hot encoded array
     */
    private static double[] oneHotEncode(int digit) {
        double[] encoding = new double[OUTPUT_SIZE];
        encoding[digit] = 1.0;
        return encoding;
    }
    
    /**
     * Find index of maximum value in array
     */
    private static int argMax(double[] array) {
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
}
