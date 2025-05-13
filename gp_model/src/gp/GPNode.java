package gp;

import java.util.*;

public class GPNode {
    public enum NodeType { FUNCTION, TERMINAL }

    public String value;
    public NodeType type;
    public List<GPNode> children;

    public GPNode(String value, NodeType type) {
        this.value = value;
        this.type = type;
        this.children = new ArrayList<>();
    }

    public GPNode clone() {
        GPNode clone = new GPNode(this.value, this.type);
        for (GPNode child : this.children) {
            clone.children.add(child.clone());
        }
        return clone;
    }

    public double evaluate(double[] input) {
        switch (type) {
            case TERMINAL:
                if (value.startsWith("x")) {
                    int index = Integer.parseInt(value.substring(1));
                    return input[index];
                } else {
                    return Double.parseDouble(value);
                }

            case FUNCTION:
                double a = children.get(0).evaluate(input);
                double b = children.get(1).evaluate(input);
                switch (value) {
                    case "+": return a + b;
                    case "-": return a - b;
                    case "*": return a * b;
                    case "/": return (Math.abs(b) < 1e-6) ? a : a / b;
                    default: throw new RuntimeException("Unknown function: " + value);
                }
        }
        return 0.0;
    }

    public String toString() {
        if (type == NodeType.TERMINAL) {
            return value;
        } else {
            return "(" + children.get(0).toString() + " " + value + " " + children.get(1).toString() + ")";
        }
    }
}