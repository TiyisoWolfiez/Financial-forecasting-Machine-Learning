package gp;

public class TreePrinter {
    public static void printTree(GPNode node) {
        printTree(node, "", true);
    }

    private static void printTree(GPNode node, String prefix, boolean isTail) {
        System.out.println(prefix + (isTail ? "└── " : "├── ") + node.value);
        for (int i = 0; i < node.children.size(); i++) {
            printTree(node.children.get(i), prefix + (isTail ? "    " : "│   "), i == node.children.size() - 1);
        }
    }
}
