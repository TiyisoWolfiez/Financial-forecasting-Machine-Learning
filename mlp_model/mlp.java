import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class mlp {
    public static void main(String[] args) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

            // Prompt for seed and file paths
            System.out.print("Enter seed value (integer): ");
            int seed = Integer.parseInt(reader.readLine());

            System.out.print("Enter path to training file (CSV): ");
            String trainPath = reader.readLine().trim();

            System.out.print("Enter path to test file (CSV): ");
            String testPath = reader.readLine().trim();

            // Load the data
            Instances trainData = loadData(trainPath);
            Instances testData = loadData(testPath);

            // Set class index (last column)
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);

            // Convert numeric class to nominal
            trainData = convertClassToNominal(trainData);
            testData = convertClassToNominal(testData);

            // Build MLP model
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setSeed(seed);
            mlp.setHiddenLayers("10,10");
            mlp.setLearningRate(0.3);
            mlp.setTrainingTime(500);

            mlp.buildClassifier(trainData);

            // === Evaluate on training data ===
            Evaluation evalTrain = new Evaluation(trainData);
            evalTrain.evaluateModel(mlp, trainData);

            // === Evaluate on test data ===
            Evaluation evalTest = new Evaluation(trainData);  // Use training data for header consistency
            evalTest.evaluateModel(mlp, testData);

            // Output results
            System.out.println("=== MLP Classification Report ===");

            System.out.println("\n--- Training Set Performance ---");
            System.out.printf("Accuracy: %.4f\n", (1 - evalTrain.errorRate()));
            System.out.printf("F1 Score: %.4f\n", evalTrain.weightedFMeasure());

            System.out.println("\n--- Testing Set Performance ---");
            System.out.printf("Accuracy: %.4f\n", (1 - evalTest.errorRate()));
            System.out.printf("F1 Score: %.4f\n", evalTest.weightedFMeasure());

        } catch (Exception e) {
            System.err.println("An error occurred:");
            e.printStackTrace();
        }
    }

    private static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        return source.getDataSet();
    }

    private static Instances convertClassToNominal(Instances data) throws Exception {
        if (data.classAttribute().isNumeric()) {
            NumericToNominal filter = new NumericToNominal();
            filter.setAttributeIndices("last");
            filter.setInputFormat(data);
            data = Filter.useFilter(data, filter);
        }
        return data;
    }
}
