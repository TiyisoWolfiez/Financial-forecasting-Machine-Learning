# Financial Forecasting with Genetic Programming

This project applies **Genetic Programming (GP)** to the problem of financial market prediction. Using a symbolic regression approach, the system evolves mathematical expressions that aim to classify financial market movements based on input features (e.g., normalized price indicators).

## Project Overview
* **Goal**: Use genetic programming to evolve programs that predict financial outcomes.
* **Input**: Preprocessed, normalized time-series financial data.
* **Output**: An evolved expression (tree) that classifies each instance.
* **Evaluation**: Accuracy on unseen test data.

## Folder Structure

```
Financial-forecasting-Machine-Learning/
├── data/
│   ├── BTC_train.csv
│   └── BTC_test.csv
├── gp_model/
│   ├── src/
│   │   └── gp/
│   │       ├── Dataset.java
│   │       ├── Node.java
│   │       ├── FunctionNode.java
│   │       ├── TerminalNode.java
│   │       ├── Individual.java
│   │       ├── GeneticProgramming.java
│   │       └── Main.java
│   └── build/
└── README.md
```

## How to Build & Run

1. **Compile the project**:
```bash
cd gp_model
```
```bash
javac -d build src/gp/*.java
```

2. **Run the program**:
```bash
java -cp build gp.Main
```

You will be prompted to:
* Enter a **random seed** (e.g., `1108`)
* Enter path to **training dataset** (e.g., `../data/BTC_train.csv`)
* Enter path to **test dataset** (e.g., `../data/BTC_test.csv`)

## Dataset Format

The dataset should be a **tab-separated** `.csv` file (`.tsv`) with no missing values.

**Example:**
```
Open	High	Low	Close	Adj Close	Output
-1.104	-1.103	-1.103	-1.100	-1.460	0
-1.098	-1.094	-1.097	-1.090	-1.404	1
...
```

* The last column (`Output`) must be `0` or `1`.
* All other columns are numeric features.

## GP Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population size | 50 | Number of programs in each generation |
| Max tree depth | 5 | Prevents overly complex expressions |
| Generations | 30 | Evolution cycles |
| Mutation rate | 0.3 | Chance of random mutation |
| Crossover rate | 0.9 | Recombination chance |
| Tournament size | 5 | Selection pressure |
| Initialization | Half-and-Half Ramp | Prevents bloat, encourages diversity |

## Sample Output

```
Generation 29: Best fitness = 0.145
Best Tree: ((((x3 + 1.0) - x1) / x0) + ... )
Test Accuracy: 68.82%
```

The best tree is a symbolic expression composed of operations over features `x0`, `x1`, ..., representing the inputs (e.g., Open, High, Low, Close, Adj Close).

## Next Steps
* Add bloat control (penalize overly complex trees)
* Export best tree as executable expression
* Compare with traditional ML models

## Author
* **Tiyiso**
* Final year Computer Science student
* COS314 - Artificial Intelligence Assignment 3