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
        convert.setAttributeIndices("last"); 
        convert.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, convert);
        testData = Filter.useFilter(testData, convert);

        J48 tree = new J48();
        tree.setUnpruned(false);     
        tree.setConfidenceFactor(0.15f); 
        tree.setMinNumObj(10); 
        tree.buildClassifier(trainData);

        // Evaluate
        Evaluation evalTrain = new Evaluation(trainData);
        evalTrain.evaluateModel(tree, trainData);
        double accTrain = evalTrain.pctCorrect() / 100.0;
        double f1Train = evalTrain.weightedFMeasure();

        Evaluation evalTest = new Evaluation(trainData);
        evalTest.evaluateModel(tree, testData);
        double accTest = evalTest.pctCorrect() / 100.0;
        double f1Test = evalTest.weightedFMeasure();

        System.out.println("==== Decision Tree Results ====");
        System.out.println("| **Model**       | **Seed value** | **Training Acc** | **Training F1** | **Testing Acc** | **Testing F1** |");
        System.out.println("|-----------------|----------------|------------------|------------------|------------------|----------------|");
        System.out.printf("| Decision Tree   | %-14d | %-16.4f | %-16.4f | %-16.4f | %-14.4f |\n", seed, accTrain, f1Train, accTest, f1Test);

        String csvContent = "Model,Seed value,Training Acc,Training F1,Testing Acc,Testing F1\n";
        csvContent += String.format("Decision Tree,%d,%.4f,%.4f,%.4f,%.4f\n",
                seed, accTrain, f1Train, accTest, f1Test);

        try (FileWriter writer = new FileWriter("Formatted_DecisionTree_Results.csv")) {
            writer.write(csvContent);
        }

        System.out.println("\nResults saved to Formatted_DecisionTree_Results.csv");
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
