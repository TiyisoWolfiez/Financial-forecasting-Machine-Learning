package src;


import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;

import java.util.Random;

public class Decision_Tree_Classifier {
    public static void main(String[] args) throws Exception {
        // Load datasets
        Instances trainData = DataSource.read("../data/BTC_train.csv");
        Instances testData = DataSource.read("../data/BTC_test.csv");

        NumericToNominal convert = new NumericToNominal();
        convert.setAttributeIndices("last");
        convert.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, convert);
        testData = Filter.useFilter(testData, convert);

        // Set class index (assumed last column is the label)
        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);

        // Create J48 Decision Tree Classifier
        J48 tree = new J48();
        tree.setUnpruned(false);            // Enable pruning
        tree.setConfidenceFactor(0.25f);    // Set pruning confidence
        tree.setMinNumObj(2);               // Minimum instances per leaf

        // Train the classifier
        tree.buildClassifier(trainData);

        // Evaluate model on test set
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(tree, testData);

        // Output performance
        System.out.println("=== J48 Decision Tree Evaluation ===");
        System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
        System.out.println("F1 Score: " + String.format("%.4f", eval.fMeasure(1))); // class index 1: "Yes"
        System.out.println(eval.toSummaryString());
        System.out.println(tree.toSummaryString());

        // Confusion matrix
        double[][] cm = eval.confusionMatrix();
        System.out.println("Confusion Matrix:");
        for (double[] row : cm) {
            for (double val : row) {
                System.out.print((int) val + " ");
            }
            System.out.println();
        }
    }
}
