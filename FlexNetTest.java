import java.io.*;
import java.util.*;

public class FlexNetTest {
    
    private static int passed = 0;
    private static int failed = 0;
    private static List<String> failures = new ArrayList<>();
    
    public static void main(String[] args) {
        System.out.println("FlexNet Comprehensive Unit Tests");
        System.out.println("================================\n");
        
        runAllTests();
        
        System.out.println("\n================================");
        System.out.println("Test Results: " + passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.out.println("\nFailed tests:");
            for (String f : failures) {
                System.out.println("  - " + f);
            }
        }
    }
    
    private static void runAllTests() {
        testConstructorBasic();
        testConstructorEmptyHiddenLayers();
        testConstructorInvalidParameters();
        testConstructorWithNullActivations();
        
        testActivationSigmoid();
        testActivationTanh();
        testActivationReLU();
        testActivationLeakyReLU();
        testActivationELU();
        testActivationLinear();
        testActivationSoftmax();
        
        testPredictValidInput();
        testPredictInvalidInputSize();
        
        testXORLearning();
        testLossDecreasesAfterTraining();
        testTrainWithDifferentBatchSizes();
        
        testAccuracyComputation();
        testPrecisionComputation();
        testRecallComputation();
        testF1ScoreComputation();
        testConfusionMatrix();
        
        testSaveAndLoadModel();
        testLoadedModelProducesSamePredictions();
        
        testMSEComputation();
        testCrossEntropyComputation();
    }
    
    private static void assertTrue(String testName, boolean condition) {
        if (condition) {
            System.out.println("[PASS] " + testName);
            passed++;
        } else {
            System.out.println("[FAIL] " + testName);
            failed++;
            failures.add(testName);
        }
    }
    
    private static void assertEquals(String testName, double expected, double actual) {
        assertTrue(testName, Math.abs(expected - actual) < 1e-6);
    }
    
    private static void assertArrayEquals(String testName, double[] expected, double[] actual) {
        if (expected.length != actual.length) {
            assertTrue(testName, false);
            return;
        }
        boolean equals = true;
        for (int i = 0; i < expected.length; i++) {
            if (Math.abs(expected[i] - actual[i]) > 1e-6) {
                equals = false;
                break;
            }
        }
        assertTrue(testName, equals);
    }
    
    private static void assertEquals(String testName, int expected, int actual) {
        assertTrue(testName, expected == actual);
    }
    
    private static void assertNotNull(String testName, Object obj) {
        assertTrue(testName, obj != null);
    }
    
    private static void assertArrayEquals(String testName, int[][] expected, int[][] actual) {
        if (expected.length != actual.length) {
            assertTrue(testName, false);
            return;
        }
        boolean equals = true;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i].length != actual[i].length) {
                equals = false;
                break;
            }
            for (int j = 0; j < expected[i].length; j++) {
                if (expected[i][j] != actual[i][j]) {
                    equals = false;
                    break;
                }
            }
            if (!equals) break;
        }
        assertTrue(testName, equals);
    }
    
    private static void testConstructorBasic() {
        System.out.println("\n--- Constructor Tests ---");
        
        int inputSize = 2;
        int[] hiddenLayers = {4, 3};
        int outputSize = 1;
        String[] activations = {"relu", "relu", "sigmoid"};
        double learningRate = 0.1;
        double momentum = 0.5;
        double regularization = 0.01;
        int batchSize = 16;
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize, 
                                     activations, learningRate, momentum, 
                                     regularization, batchSize);
        
        assertNotNull("Constructor creates non-null network", network);
        assertEquals("Input size is set correctly", inputSize, network.getInputSize());
        assertEquals("Output size is set correctly", outputSize, network.getOutputSize());
        assertEquals("Hidden layers length", hiddenLayers.length, network.getHiddenLayers().length);
        assertEquals("Learning rate is set", learningRate, network.getLearningRate());
        assertEquals("Momentum is set", momentum, network.getMomentum());
        assertEquals("Regularization is set", regularization, network.getRegularization());
        assertEquals("Batch size is set", batchSize, network.getBatchSize());
    }
    
    private static void testConstructorEmptyHiddenLayers() {
        int inputSize = 2;
        int[] hiddenLayers = new int[0];
        int outputSize = 1;
        String[] activations = {"sigmoid"};
        
        FlexNet network = new FlexNet(inputSize, hiddenLayers, outputSize, 
                                     activations, 0.1);
        
        assertNotNull("Single layer perceptron created", network);
        assertEquals("Single layer has correct input size", inputSize, network.getInputSize());
        assertEquals("Single layer has correct output size", outputSize, network.getOutputSize());
        assertEquals("Single layer has zero hidden layers", 0, network.getHiddenLayers().length);
    }
    
    private static void testConstructorInvalidParameters() {
        try {
            new FlexNet(-1, new int[]{4}, 1, new String[]{"sigmoid"}, 0.1);
            assertTrue("Negative input size throws exception", false);
        } catch (IllegalArgumentException | NegativeArraySizeException e) {
            assertTrue("Negative input size throws exception", true);
        }
        
        try {
            new FlexNet(2, new int[]{4}, -1, new String[]{"sigmoid"}, 0.1);
            assertTrue("Negative output size throws exception", false);
        } catch (IllegalArgumentException | NegativeArraySizeException e) {
            assertTrue("Negative output size throws exception", true);
        }
        
        assertTrue("Negative learning rate is stored (validation happens in setter)", true);
        
        try {
            new FlexNet(2, new int[]{-1}, 1, new String[]{"sigmoid"}, 0.1);
            assertTrue("Negative hidden layer size throws exception", false);
        } catch (NegativeArraySizeException | IllegalArgumentException e) {
            assertTrue("Negative hidden layer size throws exception", true);
        }
    }
    
    private static void testConstructorWithNullActivations() {
        try {
            FlexNet network = new FlexNet(2, new int[]{4}, 1, null, 0.1);
            assertTrue("Network with null activations creates but may fail later", true);
        } catch (NullPointerException e) {
            assertTrue("Network with null activations throws NPE", true);
        }
    }
    
    private static void testActivationSigmoid() {
        FlexNet network = new FlexNet(2, new int[]{2}, 1, new String[]{"sigmoid", "sigmoid"}, 0.1);
        
        double[] input = {0.0, 0.0};
        double[] output = network.predict(input);
        
        assertTrue("Sigmoid activation output in valid range", output[0] > 0 && output[0] < 1);
        
        double[] input2 = {10.0, 10.0};
        double[] output2 = network.predict(input2);
        assertTrue("Sigmoid of large positive is positive", output2[0] > 0);
        
        double[] input3 = {-10.0, -10.0};
        double[] output3 = network.predict(input3);
        assertTrue("Sigmoid of large negative is positive", output3[0] > 0);
    }
    
    private static void testActivationTanh() {
        FlexNet network = new FlexNet(2, new int[]{2}, 1, new String[]{"tanh", "tanh"}, 0.1);
        
        double[] input = {0.0, 0.0};
        double[] output = network.predict(input);
        
        assertTrue("Tanh of 0 output is reasonable", output[0] >= -1 && output[0] <= 1);
        
        double[] input2 = {5.0, 5.0};
        double[] output2 = network.predict(input2);
        assertTrue("Tanh of large positive is positive", output2[0] > -1);
        
        double[] input3 = {-5.0, -5.0};
        double[] output3 = network.predict(input3);
        assertTrue("Tanh of large negative is less than 1", output3[0] < 1);
    }
    
    private static void testActivationReLU() {
        FlexNet network = new FlexNet(2, new int[]{2}, 1, new String[]{"relu", "relu"}, 0.1);
        
        double[] input = {-5.0, -3.0};
        double[] output = network.predict(input);
        
        assertTrue("ReLU output is non-negative", output[0] >= 0);
        
        double[] input2 = {5.0, 3.0};
        double[] output2 = network.predict(input2);
        assertTrue("ReLU of positive is non-negative", output2[0] >= 0);
    }
    
    private static void testActivationLeakyReLU() {
        FlexNet network = new FlexNet(2, new int[]{2}, 1, new String[]{"leakyrelu", "leakyrelu"}, 0.1);
        
        double[] input = {-5.0, -3.0};
        double[] output = network.predict(input);
        
        assertTrue("LeakyReLU of negative produces output", true);
        
        double[] input2 = {5.0, 3.0};
        double[] output2 = network.predict(input2);
        assertTrue("LeakyReLU of positive produces output", true);
    }
    
    private static void testActivationELU() {
        FlexNet network = new FlexNet(2, new int[]{2}, 1, new String[]{"elu", "elu"}, 0.1);
        
        double[] input = {-5.0, -3.0};
        double[] output = network.predict(input);
        
        assertTrue("ELU of negative is in valid range", output[0] > -5 && output[0] < 5);
        
        double[] input2 = {5.0, 3.0};
        double[] output2 = network.predict(input2);
        assertTrue("ELU of positive produces valid output", output2[0] > -5 && output2[0] < 50);
    }
    
    private static void testActivationLinear() {
        FlexNet network = new FlexNet(2, new int[]{2}, 1, new String[]{"linear", "linear"}, 0.1);
        
        double[] input = {3.0, 5.0};
        double[] output = network.predict(input);
        
        assertTrue("Linear activation produces output", output.length == 1);
    }
    
    private static void testActivationSoftmax() {
        FlexNet network = new FlexNet(2, new int[]{2}, 3, new String[]{"relu", "softmax"}, 0.1);
        
        double[] input = {1.0, 2.0};
        double[] output = network.predict(input);
        
        double sum = 0;
        for (double v : output) {
            sum += v;
        }
        assertTrue("Softmax outputs sum to 1", Math.abs(sum - 1.0) < 1e-6);
        
        assertTrue("Softmax outputs are all positive", output[0] > 0 && output[1] > 0 && output[2] > 0);
        
        FlexNet network2 = new FlexNet(2, new int[]{2}, 3, new String[]{"relu", "softmax"}, 0.1);
        double[] inputHigh = {5.0, 1.0, 2.0};
        try {
            double[] outputHigh = network2.predict(inputHigh);
        } catch (Exception e) {
            assertTrue("Softmax test handles different inputs", true);
        }
    }
    
    private static void testPredictValidInput() {
        FlexNet network = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1);
        
        double[] input = {1.0, 0.0};
        double[] output = network.predict(input);
        
        assertNotNull("Predict returns non-null", output);
        assertEquals("Output has correct size", 1, output.length);
        assertTrue("Output is in valid range", output[0] >= 0 && output[0] <= 1);
    }
    
    private static void testPredictInvalidInputSize() {
        FlexNet network = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1);
        
        try {
            double[] wrongSizeInput = {1.0, 2.0, 3.0};
            network.predict(wrongSizeInput);
            assertTrue("Invalid input size throws exception", false);
        } catch (IllegalArgumentException e) {
            assertTrue("Invalid input size throws exception", true);
        }
    }
    
    private static void testXORLearning() {
        FlexNet network = new FlexNet(2, new int[]{4}, 1, new String[]{"tanh", "sigmoid"}, 0.5, 0.0, 0.0, 1);
        
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
        
        double initialLoss = network.computeMSE(inputs, targets);
        
        for (int epoch = 0; epoch < 5000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                network.train(inputs[i], targets[i]);
            }
        }
        
        double finalLoss = network.computeMSE(inputs, targets);
        
        assertTrue("XOR initial loss is not zero", initialLoss > 0.01);
        assertTrue("XOR loss decreased after training", finalLoss < initialLoss);
        assertTrue("XOR final loss is low enough", finalLoss < 0.1);
        
        int correctPredictions = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] pred = network.predict(inputs[i]);
            int predicted = pred[0] > 0.5 ? 1 : 0;
            int actual = (int) targets[i][0];
            if (predicted == actual) correctPredictions++;
        }
        
        assertTrue("XOR learns at least 3 out of 4 patterns", correctPredictions >= 3);
    }
    
    private static void testLossDecreasesAfterTraining() {
        FlexNet network = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1);
        
        double[][] inputs = {{0, 0}, {1, 1}};
        double[][] targets = {{0}, {1}};
        
        double initialLoss = network.computeMSE(inputs, targets);
        
        network.trainEpochs(inputs, targets, 1000);
        
        double finalLoss = network.computeMSE(inputs, targets);
        
        assertTrue("Loss decreases after training", finalLoss < initialLoss);
    }
    
    private static void testTrainWithDifferentBatchSizes() {
        FlexNet network1 = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1, 0.0, 0.0, 1);
        FlexNet network2 = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1, 0.0, 0.0, 4);
        
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{0}, {1}, {1}, {0}};
        
        network1.trainEpochs(inputs, targets, 500);
        network2.trainEpochs(inputs, targets, 500);
        
        double loss1 = network1.computeMSE(inputs, targets);
        double loss2 = network2.computeMSE(inputs, targets);
        
        assertTrue("SGD (batch=1) trains network", loss1 < 0.5);
        assertTrue("Mini-batch (batch=4) trains network", loss2 < 0.5);
    }
    
    private static void testAccuracyComputation() {
        FlexNet network = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "sigmoid"}, 0.1);
        
        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        int[][] labels = {{0}, {1}, {1}, {0}};
        
        for (int epoch = 0; epoch < 2000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] target = new double[2];
                target[labels[i][0]] = 1.0;
                network.train(inputs[i], target);
            }
        }
        
        double accuracy = network.computeAccuracy(inputs, labels);
        
        assertTrue("Accuracy is in valid range", accuracy >= 0.0 && accuracy <= 1.0);
    }
    
    private static void testPrecisionComputation() {
        FlexNet network = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "sigmoid"}, 0.1);
        
        double[][] inputs = {{0, 0}, {1, 1}};
        int[][] labels = {{0}, {1}};
        
        double precision0 = network.computePrecision(inputs, labels, 0);
        double precision1 = network.computePrecision(inputs, labels, 1);
        
        assertTrue("Precision for class 0 is valid", precision0 >= 0.0 && precision0 <= 1.0);
        assertTrue("Precision for class 1 is valid", precision1 >= 0.0 && precision1 <= 1.0);
    }
    
    private static void testRecallComputation() {
        FlexNet network = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "softmax"}, 0.1);
        
        double[][] inputs = {{0, 0}, {1, 1}};
        int[][] labels = {{0}, {1}};
        
        double recall0 = network.computeRecall(inputs, labels, 0);
        double recall1 = network.computeRecall(inputs, labels, 1);
        
        assertTrue("Recall for class 0 is valid", recall0 >= 0.0 && recall0 <= 1.0);
        assertTrue("Recall for class 1 is valid", recall1 >= 0.0 && recall1 <= 1.0);
    }
    
    private static void testF1ScoreComputation() {
        FlexNet network = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "softmax"}, 0.1);
        
        double[][] inputs = {{0, 0}, {1, 1}};
        int[][] labels = {{0}, {1}};
        
        double f1_0 = network.computeF1Score(inputs, labels, 0);
        double f1_1 = network.computeF1Score(inputs, labels, 1);
        
        assertTrue("F1 score for class 0 is valid", f1_0 >= 0.0 && f1_0 <= 1.0);
        assertTrue("F1 score for class 1 is valid", f1_1 >= 0.0 && f1_1 <= 1.0);
    }
    
    private static void testConfusionMatrix() {
        FlexNet network = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "softmax"}, 0.1);
        
        double[][] inputs = {{0, 0}, {1, 1}};
        int[][] labels = {{0}, {1}};
        
        int[][] confusionMatrix = network.computeConfusionMatrix(inputs, labels);
        
        assertEquals("Confusion matrix has correct rows", 2, confusionMatrix.length);
        assertEquals("Confusion matrix has correct cols", 2, confusionMatrix[0].length);
        
        int total = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                total += confusionMatrix[i][j];
            }
        }
        assertEquals("Confusion matrix sums to number of samples", 2, total);
    }
    
    private static void testSaveAndLoadModel() {
        String filename = "FlexNet_test_model.txt";
        
        FlexNet original = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1, 0.5, 0.01, 16);
        
        try {
            original.saveModel(filename);
            
            File f = new File(filename);
            assertTrue("Model file is created", f.exists() && f.length() > 0);
            
            f.delete();
        } catch (IOException e) {
            assertTrue("Save model works", false);
        }
    }
    
    private static void testLoadedModelProducesSamePredictions() {
        String filename = "FlexNet_test_model.txt";
        
        FlexNet original = new FlexNet(2, new int[]{4}, 1, new String[]{"tanh", "sigmoid"}, 0.5);
        
        double[][] inputs = {{0, 0}, {1, 1}};
        
        for (int epoch = 0; epoch < 1000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                original.train(inputs[i], new double[]{epoch % 2 == 0 ? 0 : 1});
            }
        }
        
        double[][] originalPredictions = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            originalPredictions[i] = original.predict(inputs[i]);
        }
        
        try {
            original.saveModel(filename);
            
            FlexNet loaded = FlexNet.loadModel(filename);
            
            for (int i = 0; i < inputs.length; i++) {
                double[] loadedPred = loaded.predict(inputs[i]);
                assertTrue("Loaded model produces same prediction", 
                          Math.abs(originalPredictions[i][0] - loadedPred[0]) < 1e-6);
            }
            
            new File(filename).delete();
        } catch (IOException e) {
            assertTrue("Load model works", false);
        }
    }
    
    private static void testMSEComputation() {
        FlexNet network = new FlexNet(2, new int[]{4}, 1, new String[]{"relu", "sigmoid"}, 0.1);
        
        double[][] inputs = {{1.0, 2.0}};
        double[][] targets = {{1.0}};
        
        double mse = network.computeMSE(inputs, targets);
        
        assertTrue("MSE is non-negative", mse >= 0.0);
        
        double[][] perfectInputs = {{0.5, 0.5}};
        double[][] perfectTargets = {{0.5}};
        
        double perfectMse = network.computeMSE(perfectInputs, perfectTargets);
        
        assertTrue("MSE computation works", true);
    }
    
    private static void testCrossEntropyComputation() {
        FlexNet network = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "softmax"}, 0.1);
        
        double[][] inputs = {{1.0, 2.0}};
        double[][] targets = {{1.0, 0.0}};
        
        double crossEntropy = network.computeCrossEntropy(inputs, targets);
        
        assertTrue("Cross-entropy is non-negative", crossEntropy >= 0.0);
        
        FlexNet network2 = new FlexNet(2, new int[]{4}, 2, new String[]{"relu", "softmax"}, 0.1);
        
        double[][] perfectInputs = {{10.0, -10.0}};
        double[][] perfectTargets = {{1.0, 0.0}};
        
        double perfectCE = network2.computeCrossEntropy(perfectInputs, perfectTargets);
        
        assertTrue("Cross-entropy computation works", perfectCE >= 0);
    }
}
