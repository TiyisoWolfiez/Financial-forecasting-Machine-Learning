package gp;

import java.util.*;

public class GPAlgorithm {
    private Dataset trainData;
    private Random rand;
    private int populationSize;
    private int maxDepth;
    private int generations;
    private double mutationRate;
    private double crossoverRate;
    private int tournamentSize;

    private List<Individual> population;

    public GPAlgorithm(Dataset trainData, int seed, int populationSize, int maxDepth, int generations,
                       double mutationRate, double crossoverRate, int tournamentSize) {
        this.trainData = trainData;
        this.rand = new Random(seed);
        this.populationSize = populationSize;
        this.maxDepth = maxDepth;
        this.generations = generations;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
        this.tournamentSize = tournamentSize;
    }

    public Individual run() {
        initializePopulation();
        evaluateFitness();

        for (int gen = 0; gen < generations; gen++) {
            List<Individual> newPop = new ArrayList<>();

            while (newPop.size() < populationSize) {
                Individual parent1 = tournamentSelection();
                Individual parent2 = tournamentSelection();

                Individual child1 = parent1.clone();
                Individual child2 = parent2.clone();

                if (rand.nextDouble() < crossoverRate) {
                    child1.tree.crossover(child2.tree);
                }

                if (rand.nextDouble() < mutationRate) {
                    child1.tree.mutate();
                }
                if (rand.nextDouble() < mutationRate) {
                    child2.tree.mutate();
                }

                newPop.add(child1);
                if (newPop.size() < populationSize)
                    newPop.add(child2);
            }

            population = newPop;
            evaluateFitness();

            // log progress
            Individual best = getBest();
            System.out.println("Generation " + gen + ": Best fitness = " + best.fitness);
        }

        return getBest();
    }

    private void initializePopulation() {
        population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            Tree tree = new Tree(rand, maxDepth, trainData.getFeatureCount());
            population.add(new Individual(tree));
        }
    }

    private void evaluateFitness() {
        double[][] X = trainData.getData();
        int[] y = trainData.getLabels();

        for (Individual ind : population) {
            int correct = 0;
            for (int i = 0; i < X.length; i++) {
                double score = ind.evaluate(X[i]);
                int predicted = score >= 0.5 ? 1 : 0;
                if (predicted == y[i]) correct++;
            }
            ind.fitness = 1.0 - (double) correct / X.length; // error = 1 - accuracy
        }
    }

    private Individual tournamentSelection() {
        Individual best = null;
        for (int i = 0; i < tournamentSize; i++) {
            Individual candidate = population.get(rand.nextInt(population.size()));
            if (best == null || candidate.fitness < best.fitness) {
                best = candidate;
            }
        }
        return best;
    }

    private Individual getBest() {
        return Collections.min(population, Comparator.comparingDouble(ind -> ind.fitness));
    }
}