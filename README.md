# COS314 Assignment 3 — Machine Learning for Financial Forecasting

This repository contains our group's implementation for Assignment 3 of COS314: Artificial Intelligence.  
The objective is to build and evaluate three ML models that classify whether a stock should be purchased based on historical data.

## File Structure

- `report/` — PDF report detailing our methodology, models, and evaluation (Wilcoxon signed-rank test).
- `data/` — Training and testing datasets (provided).
- `gp_model/` — Genetic Programming implementation in Java/C++.
- `mlp_model/` — Python implementation of Multi-Layer Perceptron (using a library like Keras or scikit-learn).
- `decision_tree/` — Weka-based J48 Decision Tree implementation.
- `results/` — Performance metrics, including accuracy and F1 score tables.

##  Models

### 1. Genetic Programming (GP)
- Language: Java
- Be on this Directory: `cd gp_model`
- Command: `javac -d build src/gp/*.java`
- Run: `java -cp build gp.Main`

### 2. Multi-Layer Perceptron (MLP)
- Language: Python
- Dependencies: Listed in `mlp_model/requirements.txt`
- Command: `python mlp.py --seed <seed> --train <train_file> --test <test_file>`

### 3. Decision Tree (J48)
- Tool: Weka
- Command: See `decision_tree/run_weka_j48.bat` for setup

##  Setup

1. Clone the repository:
   ```bash
   gh repo clone TiyisoWolfiez/Financial-forecasting-Machine-Learning
   ```
   ```bash
   cd Financial-forecasting-Machine-Learning
   ```

2. Install Python dependencies for MLP
   ```bash
   pip install -r mlp_model/requirements.txt
   ```
3. Run the respective model scripts as described above.

## Group Mambers
  - Member 1 : Genetic Programming
  - Member 2: Multi-Layer Perceptron
  - Member 3: Decision Tree
  - Member 4: Report Writer

## Notes
 - Ensure all models prompt for `seed` and file paths at runtime.
