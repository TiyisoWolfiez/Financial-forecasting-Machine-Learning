package gp;

import java.util.*;

public class Tree {
    public GPNode root;
    private static final String[] FUNCTIONS = { "+", "-", "*", "/" };
    private static final String[] CONSTANTS = { "1.0", "0.5", "-1.0", "2.0" };
    private Random rand;
    private int maxDepth;
    private int numFeatures;

    public Tree(Random rand, int maxDepth, int numFeatures) {
        this.rand = rand;
        this.maxDepth = maxDepth;
        this.numFeatures = numFeatures;
        this.root = generateRandomTree(0);
    }

    private GPNode generateRandomTree(int depth) {
        boolean isTerminal = depth >= maxDepth || rand.nextDouble() < 0.3;
        if (isTerminal) {
            if (rand.nextBoolean()) {
                // Feature input
                String feature = "x" + rand.nextInt(numFeatures);
                return new GPNode(feature, GPNode.NodeType.TERMINAL);
            } else {
                // Constant
                String constant = CONSTANTS[rand.nextInt(CONSTANTS.length)];
                return new GPNode(constant, GPNode.NodeType.TERMINAL);
            }
        } else {
            // Function node
            String function = FUNCTIONS[rand.nextInt(FUNCTIONS.length)];
            GPNode node = new GPNode(function, GPNode.NodeType.FUNCTION);
            node.children.add(generateRandomTree(depth + 1));
            node.children.add(generateRandomTree(depth + 1));
            return node;
        }
    }

    public double evaluate(double[] input) {
        return root.evaluate(input);
    }

    public Tree clone() {
        Tree copy = new Tree(rand, maxDepth, numFeatures);
        copy.root = this.root.clone();
        return copy;
    }

    public void mutate() {
        List<GPNode> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        GPNode toReplace = nodes.get(rand.nextInt(nodes.size()));
        GPNode replacement = generateRandomTree(0);
        replaceSubtree(root, toReplace, replacement);
    }

    public void crossover(Tree other) {
        List<GPNode> thisNodes = new ArrayList<>();
        List<GPNode> otherNodes = new ArrayList<>();
        collectNodes(this.root, thisNodes);
        collectNodes(other.root, otherNodes);

        GPNode a = thisNodes.get(rand.nextInt(thisNodes.size()));
        GPNode b = otherNodes.get(rand.nextInt(otherNodes.size()));

        GPNode temp = a.clone();
        replaceSubtree(this.root, a, b.clone());
        replaceSubtree(other.root, b, temp);
    }

    private void collectNodes(GPNode node, List<GPNode> list) {
        list.add(node);
        for (GPNode child : node.children) {
            collectNodes(child, list);
        }
    }

    private boolean replaceSubtree(GPNode current, GPNode target, GPNode replacement) {
        for (int i = 0; i < current.children.size(); i++) {
            if (current.children.get(i) == target) {
                current.children.set(i, replacement);
                return true;
            } else if (replaceSubtree(current.children.get(i), target, replacement)) {
                return true;
            }
        }
        return false;
    }

    public String toString() {
        return root.toString();
    }
}