package gp;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {

    public static void main(String[] args) {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

            System.out.print("Enter seed: ");
            int seed = Integer.parseInt(br.readLine());

            System.out.print("Enter path to training set (e.g., ../data/BTC_train.csv): ");
            String trainPath = br.readLine();

            System.out.print("Enter path to test set (e.g., ../data/BTC_test.csv): ");
            String testPath = br.readLine();

            System.out.print("Enter population size: ");
            int populationSize = Integer.parseInt(br.readLine());

            System.out.print("Enter max tree depth: ");
            int maxDepth = Integer.parseInt(br.readLine());

            System.out.print("Enter number of generations: ");
            int generations = Integer.parseInt(br.readLine());

            System.out.print("Enter mutation rate (e.g., 0.3): ");
            double mutationRate = Double.parseDouble(br.readLine());

            System.out.print("Enter crossover rate (e.g., 0.9): ");
            double crossoverRate = Double.parseDouble(br.readLine());

            System.out.print("Enter tournament size: ");
            int tournamentSize = Integer.parseInt(br.readLine());

            Dataset train = new Dataset(trainPath);
            Dataset test = new Dataset(testPath);

            GPAlgorithm gp = new GPAlgorithm(
                train,
                seed,
                populationSize,
                maxDepth,
                generations,
                mutationRate,
                crossoverRate,
                tournamentSize
            );

            Individual best = gp.run();

            System.out.println("\nBest Tree:");
            System.out.println(best.tree.toString());

            System.out.println("\nBest Tree (Visual Representation):");
            TreePrinter.printTree(best.tree.root);

            System.out.println("\n---------------------------------- Evaluation -----------------------------------");

            double[] trainMetrics = evaluate(best, train);
            double[] testMetrics = evaluate(best, test);

            // Print final results table
            System.out.println("\n====================================== RESULTS TABLE ====================================");
            System.out.printf("| %-20s | %-10s | %-11s | %-9s | %-11s | %-9s |\n",
                    "Model", "Seed", "Train Acc", "Train F1", "Test Acc", "Test F1");
            System.out.println("-----------------------------------------------------------------------------------------");
            System.out.printf("| %-20s | %-10d | %-11.2f | %-9.4f | %-11.2f | %-9.4f |\n",
                    "Genetic Programming", seed,
                    trainMetrics[0] * 100, trainMetrics[1],
                    testMetrics[0] * 100, testMetrics[1]);
            System.out.println("=========================================================================================");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double[] evaluate(Individual model, Dataset dataset) {
        double[][] X = dataset.getData();
        int[] y = dataset.getLabels();

        int correct = 0;
        int tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < X.length; i++) {
            double score = model.evaluate(X[i]);
            int predicted = score >= 0.5 ? 1 : 0;

            if (predicted == y[i]) correct++;

            if (predicted == 1 && y[i] == 1) tp++;
            else if (predicted == 1 && y[i] == 0) fp++;
            else if (predicted == 0 && y[i] == 1) fn++;
        }

        double accuracy = (double) correct / X.length;
        double precision = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        double recall = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

        return new double[]{accuracy, f1};
    }
}