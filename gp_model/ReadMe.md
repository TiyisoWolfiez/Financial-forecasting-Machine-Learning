# 📈 Financial Forecasting with Genetic Programming

A Java-based implementation of **Genetic Programming (GP)** for financial time-series forecasting and classification. This project evolves mathematical expressions as tree structures to predict binary outcomes from numeric financial data, specifically designed for Bitcoin price movement prediction.

## 🎯 Overview

This system uses evolutionary algorithms to automatically generate and optimize mathematical models for financial forecasting. The GP approach evolves populations of expression trees that can capture complex, non-linear relationships in financial time-series data without requiring manual feature engineering.

### Key Features
- **Evolutionary Model Generation**: Automatically discovers optimal mathematical expressions
- **Binary Classification**: Predicts up/down price movements 
- **Robust Evaluation**: Comprehensive accuracy and F1-score metrics
- **Reproducible Results**: Seeded random number generation
- **Extensible Architecture**: Modular design for easy customization

---

##  Project Structure

```
Financial-forecasting-Machine-Learning/
├── data/
│   ├── BTC_train.csv              # Training dataset
│   └── BTC_test.csv               # Test dataset
├── gp_model/
│   ├── src/
│   │   └── gp/
│   │       ├── Dataset.java       # Data loading and preprocessing
│   │       ├── FunctionNode.java  # Mathematical operators (+, -, *, /)
│   │       ├── TerminalNode.java  # Variables and constants
│   │       ├── Node.java          # Abstract base class for tree nodes
│   │       ├── Individual.java    # GP individual (expression tree)
│   │       ├── GeneticProgramming.java # Main GP algorithm
│   │       └── Main.java          # Application entry point
│   ├── build/                     # Compiled .class files
│   ├── manifest.txt               # JAR manifest configuration
│   └── gp_forecaster.jar          #  Executable JAR file
└── README.md
```

---

## Requirements

- **Java Development Kit (JDK) 8+**
- **Command Line Interface** (Terminal/CMD/PowerShell)
- **CSV Data Files** with appropriate format

### Data Format
The CSV files should contain:
- Numeric features (x1, x2, x3, etc.)
- Binary target column (0 or 1)
- Header row with column names

---

## Running The Existing Jar:
### Step 1: Navigate to Project Directory
```bash
cd gp_model
```
### Step 2: Run the JAR
```bash
java -jar gp_forecaster.jar
```

## Build Instructions

### Step 1: Navigate to Project Directory
```bash
cd gp_model
```

### Step 2: Compile Java Source Code
```bash
javac -d build src/gp/*.java
```

### Step 3: Create JAR File
Ensure your `manifest.txt` contains:
```
Main-Class: gp.Main

```
*(Note: Empty line at the end is required)*

Build the JAR:
```bash
jar cfm gp_forecaster.jar manifest.txt -C build .
```

---

##  Usage

### Running the Application
```bash
java -jar gp_forecaster.jar
```

### Interactive Setup
The program will prompt for configuration:

```bash
Enter seed: 42
Enter path to training set (e.g., ../data/BTC_train.csv): ../data/BTC_train.csv
Enter path to test set (e.g., ../data/BTC_test.csv): ../data/BTC_test.csv
```

### Command Line Arguments (Alternative)
```bash
java -jar gp_forecaster.jar
```

---

##  Expected Output

### Evolution Progress
```
Generation 1: Best Fitness = 0.7234
Generation 2: Best Fitness = 0.7456
...
Generation 50: Best Fitness = 0.8437
```

### Final Results
```
==================== BEST EVOLVED MODEL ====================
Best Tree: (((x3 + 1.0) - -1.0) / x3)

======================= EVALUATION =======================
Training Set Performance:
  - Accuracy: 84.37%
  - F1 Score: 0.8221
  - Predictions: 1250/1483 correct

Test Set Performance:
  - Accuracy: 79.12%
  - F1 Score: 0.7734
  - Predictions: 395/499 correct

====================================== RESULTS TABLE ======================================
| Model               | Seed | Train Acc | Train F1 | Test Acc | Test F1 | Complexity |
----------------------------------------------------------------------------------------
| Genetic Programming | 42   | 84.37     | 0.8221   | 79.12    | 0.7734  | 7 nodes    |
==========================================================================================
```

---

##  Algorithm Parameters

The current GP configuration includes:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population Size | 100 | Number of individuals per generation |
| Generations | 50 | Maximum evolution cycles |
| Tournament Size | 5 | Selection pressure |
| Crossover Rate | 0.8 | Probability of genetic recombination |
| Mutation Rate | 0.1 | Probability of random changes |
| Max Tree Depth | 8 | Maximum expression complexity |

---