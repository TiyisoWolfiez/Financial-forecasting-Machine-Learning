package gp;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

            System.out.print("Enter seed: ");
            int seed = Integer.parseInt(br.readLine());

            System.out.print("Enter path to training set (e.g., data/BTC_train.csv): ");
            String trainPath = br.readLine();

            System.out.print("Enter path to test set (e.g., data/BTC_test.csv): ");
            String testPath = br.readLine();

            Dataset train = new Dataset(trainPath);
            Dataset test = new Dataset(testPath);

            GPAlgorithm gp = new GPAlgorithm(
                train,
                seed,
                50,     // population size
                5,      // max depth
                30,     // generations
                0.3,    // mutation rate
                0.9,    // crossover rate
                5       // tournament size
            );

            Individual best = gp.run();

            System.out.println("\nBest Tree:");
            System.out.println(best.tree.toString());

            evaluateOnTestSet(best, test);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void evaluateOnTestSet(Individual best, Dataset test) {
        double[][] X = test.getData();
        int[] y = test.getLabels();

        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            double score = best.evaluate(X[i]);
            int predicted = score >= 0.5 ? 1 : 0;
            if (predicted == y[i]) correct++;
        }

        double accuracy = (double) correct / X.length;
        System.out.println("Test Accuracy: " + String.format("%.2f%%", accuracy * 100));
    }
}