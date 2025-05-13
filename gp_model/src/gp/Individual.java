package gp;

public class Individual {
    public Tree tree;
    public double fitness;

    public Individual(Tree tree) {
        this.tree = tree;
        this.fitness = Double.MAX_VALUE;
    }

    public double evaluate(double[] input) {
        return tree.evaluate(input);
    }

    public Individual clone() {
        Individual copy = new Individual(tree.clone());
        copy.fitness = this.fitness;
        return copy;
    }

    public String toString() {
        return "Fitness: " + fitness + ", Tree: " + tree.toString();
    }
}