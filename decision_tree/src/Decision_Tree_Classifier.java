package src;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.Discretize;

import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;

import java.util.Scanner;
import java.io.FileWriter;

public class Decision_Tree_Classifier {
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);

        // Prompt user input for file paths and seed
        System.out.print("Enter path to training CSV file: ");
        String trainPath = scanner.nextLine();

        System.out.print("Enter path to testing CSV file: ");
        String testPath = scanner.nextLine();

        System.out.print("Enter random seed value: ");
        int seed = Integer.parseInt(scanner.nextLine());

        // Load and prepare data
        DataSource trainSource = new DataSource(trainPath);
        Instances trainData = trainSource.getDataSet();

        DataSource testSource = new DataSource(testPath);
        Instances testData = testSource.getDataSet();

        weka.filters.unsupervised.attribute.Normalize normalize = new weka.filters.unsupervised.attribute.Normalize();
        normalize.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, normalize);
        testData = Filter.useFilter(testData, normalize);

        Discretize discretize = new Discretize();
        discretize.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, discretize);
        testData = Filter.useFilter(testData, discretize);

        // Convert last attribute to nominal
        NumericToNominal convertTrain = new NumericToNominal();
        convertTrain.setAttributeIndices("last");
        convertTrain.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, convertTrain);

        NumericToNominal convertTest = new NumericToNominal();
        convertTest.setAttributeIndices("last");
        convertTest.setInputFormat(testData);
        testData = Filter.useFilter(testData, convertTest);

        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);

        // Build J48 classifier
        J48 tree = new J48();
        tree.setUnpruned(true);
        tree.setMinNumObj(1);
        tree.buildClassifier(trainData);

        // Evaluation
        Evaluation evalTrain = new Evaluation(trainData);
        evalTrain.evaluateModel(tree, trainData);

        Evaluation evalTest = new Evaluation(trainData);
        evalTest.evaluateModel(tree, testData);

        // Output results
        double accTrain = evalTrain.pctCorrect() / 100.0;
        double f1Train = evalTrain.weightedFMeasure();
        double accTest = evalTest.pctCorrect() / 100.0;
        double f1Test = evalTest.weightedFMeasure();

        System.out.println("=== Training Summary ===");
        System.out.println(evalTrain.toSummaryString());

        System.out.println("=== Testing Summary ===");
        System.out.println(evalTest.toSummaryString());

        System.out.println("=== Decision Tree ===");
        System.out.println(tree);

        // Save results to file
        try (FileWriter writer = new FileWriter("Formatted_DecisionTree_Results.csv")) {
            writer.write("Model,Seed,Training Accuracy,F1 Score,Testing Accuracy,F1 Score\n");
            writer.write(String.format("Decision Tree,%d,%.4f,%.4f,%.4f,%.4f\n",
                    seed, accTrain, f1Train, accTest, f1Test));
        }

        System.out.println("Results written to Formatted_DecisionTree_Results.csv");
    }
}
