# COS314 - Assignment 3: Multi-Layer Perceptron Classifier (Java)

## Description

This Java program implements a **Multi-Layer Perceptron (MLP)** neural network using the **Weka** library. It classifies whether a cryptocurrency stock (e.g., Bitcoin) should be purchased based on historical data from CSV files. The model is evaluated using **Accuracy** and **F1 Score** on both the training and testing datasets.

## Dependencies

Make sure the following are installed and available:
- Java 8 or later
- [Weka 3.9.6](https://www.cs.waikato.ac.nz/ml/weka/) (`weka.jar`)
- [MTJ (Matrix Toolkit for Java)](https://github.com/fommil/matrix-toolkits-java) (`mtj.jar`)

## Compilation

```bash
javac -cp .:weka-3-9-6/weka.jar:mtj.jar mlp.java
```

## Running the Program

### For Java 8 (recommended):
```bash
java -cp .:weka-3-9-6/weka.jar:mtj.jar mlp
```

### For Java 9 or later:
```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -cp .:weka-3-9-6/weka.jar:mtj.jar mlp
```

## Running a JAR File

1. Package into a JAR:
```bash
jar cfm mlp.jar manifest.txt mlp.class
```

2. Run the JAR:
```bash
java -jar mlp.jar
```

## File Structure

- `mlp.java` – Java MLP implementation using Weka
- `BTC_train.csv` – Training dataset
- `BTC_test.csv` – Testing dataset
- `weka.jar` – Weka library
- `mtj.jar` – Matrix operations library
- `manifest.txt` – Manifest for JAR packaging
- `mlp.jar` – Compiled runnable JAR (after build)
- `requirements.txt` – Initial setup instructions

## How to Use

1. Open terminal in the `mlp_model` directory
2. Compile and run the program using the commands above
3. When prompted, enter the following:
   - Seed value (e.g., `77`)
   - Path to training CSV file (e.g., `../data/BTC_train.csv`)
   - Path to testing CSV file (e.g., `../data/BTC_test.csv`)

## Sample Output

```
=== MLP Classification Report ===
--- Training Set Performance ---
Accuracy: 0.8912
F1 Score: 0.8945
--- Testing Set Performance ---
Accuracy: 0.7430
F1 Score: 0.7519
```