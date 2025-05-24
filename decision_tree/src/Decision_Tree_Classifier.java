package src;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;

import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;

public class Decision_Tree_Classifier {
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter path to training CSV file: ");
        String trainPath = scanner.nextLine();

        System.out.print("Enter path to testing CSV file: ");
        String testPath = scanner.nextLine();

        System.out.print("Enter random seed value: ");
        int seed = Integer.parseInt(scanner.nextLine());

        Instances trainData = new DataSource(trainPath).getDataSet();
        Instances testData = new DataSource(testPath).getDataSet();

   
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

        Discretize discretize = new Discretize();
        discretize.setAttributeIndices("first-last-1"); 
        discretize.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, discretize);
        testData = Filter.useFilter(testData, discretize);

        
        J48 tree = new J48();
        tree.setUnpruned(false);               // Enable pruning
        tree.setReducedErrorPruning(true);     // Use reduced-error pruning
        tree.setNumFolds(3);                   // Pruning folds
        tree.setSeed(seed);                    // <-- seed has an effect
        tree.setMinNumObj(1);                  // Allow small leaves
        tree.buildClassifier(trainData);

        Evaluation evalTrain = new Evaluation(trainData);
        evalTrain.evaluateModel(tree, trainData);

        Evaluation evalTest = new Evaluation(trainData);
        evalTest.evaluateModel(tree, testData);

        Evaluation evalCV = new Evaluation(trainData);
        evalCV.crossValidateModel(tree, trainData, 10, new Random(seed));


        double accTrain = evalTrain.pctCorrect() / 100.0;
        double f1Train = evalTrain.weightedFMeasure();
        double accTest = evalTest.pctCorrect() / 100.0;
        double f1Test = evalTest.weightedFMeasure();
        double accCV = evalCV.pctCorrect() / 100.0;
        double f1CV = evalCV.weightedFMeasure();


        System.out.println(tree.toString());
        System.out.println("\nConfusion Matrix:");
        for (double[] row : evalTest.confusionMatrix()) {
            for (double val : row) {
                System.out.print((int) val + "\t");
            }
            System.out.println();
        }

        System.out.println("\n==== Evaluation Summary ====");
        System.out.printf("Training Accuracy         : %.4f%%\n", accTrain * 100);
        System.out.printf("Training F1 Score         : %.4f\n", f1Train);
        System.out.printf("Testing Accuracy          : %.4f%%\n", accTest * 100);
        System.out.printf("Testing F1 Score          : %.4f\n", f1Test);
        System.out.printf("Cross-Validation Accuracy : %.4f%%\n", accCV * 100);
        System.out.printf("Cross-Validation F1 Score : %.4f\n", f1CV);


        System.out.println("\n==== Decision Tree Results ====");
        System.out.println("| **Model**       | **Seed value** | **Training Acc** | **Training F1** | **Testing Acc** | **Testing F1** |");
        System.out.println("|-----------------|----------------|------------------|------------------|------------------|----------------|");
        System.out.printf("| Decision Tree   | %-14d | %-16.4f | %-16.4f | %-16.4f | %-14.4f |\n",
                seed, accTrain, f1Train, accTest, f1Test);

        try (FileWriter writer = new FileWriter("DecisionTree_Results.csv")) {
            writer.write("Model,Seed,Training Accuracy,F1 Score,Testing Accuracy,F1 Score\n");
            writer.write(String.format("Decision Tree,%d,%.4f,%.4f,%.4f,%.4f\n",
                    seed, accTrain, f1Train, accTest, f1Test));
        }

        System.out.println("\nResults written to DecisionTree_Results.csv");
    }
}
