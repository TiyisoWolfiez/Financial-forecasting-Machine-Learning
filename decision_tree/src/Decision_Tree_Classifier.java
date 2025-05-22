package src;


import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;
import weka.core.Instance;


import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;


public class Decision_Tree_Classifier {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Usage: java src.Decision_Tree_Classifier <seed>");
            return;
        }

        int seed = Integer.parseInt(args[0]);

        // File paths (modify here if needed)
        String trainPath = "../data/BTC_train.csv";
        String testPath = "../data/BTC_test.csv";

        // Load training data
        DataSource trainSource = new DataSource(trainPath);
        Instances trainData = trainSource.getDataSet();
        if (trainData.classIndex() == -1)
            trainData.setClassIndex(trainData.numAttributes() - 1);

        // Load test data
        DataSource testSource = new DataSource(testPath);
        Instances testData = testSource.getDataSet();
        if (testData.classIndex() == -1)
            testData.setClassIndex(testData.numAttributes() - 1);

        NumericToNominal convert = new NumericToNominal();
        convert.setAttributeIndices("last"); // assuming the class attribute is the last
        convert.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, convert);
        testData = Filter.useFilter(testData, convert);

        J48 tree = new J48();
        tree.buildClassifier(trainData);

        // Evaluate
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(tree, testData);

        // Output to CSV
        try (PrintWriter writer = new PrintWriter(new FileWriter("decision_tree_results.csv"))) {
            writer.println("Instance,Actual,Predicted");
            for (int i = 0; i < testData.numInstances(); i++) {
                Instance instance = testData.instance(i);
                double actualClass = instance.classValue();
                double predictedClass = tree.classifyInstance(instance);

                String actualLabel = testData.classAttribute().value((int) actualClass);
                String predictedLabel = testData.classAttribute().value((int) predictedClass);

                writer.printf("%d,%s,%s%n", i + 1, actualLabel, predictedLabel);
            }

            writer.println();
            writer.println("=== Evaluation Summary ===");
            writer.printf("Correctly Classified Instances: %d%n", (int) eval.correct());
            writer.printf("Incorrectly Classified Instances: %d%n", (int) eval.incorrect());
            writer.printf("Accuracy: %.2f%%%n", eval.pctCorrect());

            writer.println("\n=== Confusion Matrix ===");
            double[][] confusionMatrix = eval.confusionMatrix();
            String[] classLabels = new String[testData.numClasses()];
            for (int i = 0; i < testData.numClasses(); i++) {
                classLabels[i] = testData.classAttribute().value(i);
            }

            writer.print("Actual \\ Predicted");
            for (String label : classLabels) {
                writer.print("," + label);
            }
            writer.println();

            for (int i = 0; i < confusionMatrix.length; i++) {
                writer.print(classLabels[i]);
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    writer.printf(",%.0f", confusionMatrix[i][j]);
                }
                writer.println();
            }
        }

        System.out.println("Evaluation complete. Results (with confusion matrix) saved to 'decision_tree_results.csv'.");
    }

        // Load datasets
    //     Instances trainData = DataSource.read("../data/BTC_train.csv");
    //     Instances testData = DataSource.read("../data/BTC_test.csv");

    //     NumericToNominal convert = new NumericToNominal();
    //     convert.setAttributeIndices("last");
    //     convert.setInputFormat(trainData);
    //     trainData = Filter.useFilter(trainData, convert);
    //     testData = Filter.useFilter(testData, convert);

    //     // Set class index (assumed last column is the label)
    //     trainData.setClassIndex(trainData.numAttributes() - 1);
    //     testData.setClassIndex(testData.numAttributes() - 1);

    //     // Create J48 Decision Tree Classifier
    //     J48 tree = new J48();
    //     tree.setUnpruned(false);            // Enable pruning
    //     tree.setConfidenceFactor(0.25f);    // Set pruning confidence
    //     tree.setMinNumObj(2);               // Minimum instances per leaf

    //     // Train the classifier
    //     tree.buildClassifier(trainData);

    //     // Evaluate model on test set
    //     Evaluation eval = new Evaluation(trainData);
    //     eval.evaluateModel(tree, testData);

    //     // Output performance
    //     System.out.println("=== J48 Decision Tree Evaluation ===");
    //     System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
    //     System.out.println("F1 Score: " + String.format("%.4f", eval.fMeasure(1))); // class index 1: "Yes"
    //     System.out.println(eval.toSummaryString());
    //     System.out.println(tree.toSummaryString());

    //     // Confusion matrix
    //     double[][] cm = eval.confusionMatrix();
    //     System.out.println("Confusion Matrix:");
    //     for (double[] row : cm) {
    //         for (double val : row) {
    //             System.out.print((int) val + " ");
    //         }
    //         System.out.println();
    //     }
    // }
}
