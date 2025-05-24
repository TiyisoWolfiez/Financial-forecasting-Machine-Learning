# COS314 Assignment 3 — Machine Learning for Financial Forecasting

This repository contains our group's implementation for Assignment 3 of COS314: Artificial Intelligence.  
The objective is to build and evaluate three ML models that classify whether a stock should be purchased based on historical data.

## File Structure

- `report/` — PDF report detailing our methodology, models, and evaluation (Wilcoxon signed-rank test).
- `data/` — Training and testing datasets (provided).
- `gp_model/` — Genetic Programming implementation in Java.
- `mlp_model/` — implementation of Multi-Layer Perceptron.
- `decision_tree/` — Weka-based J48 Decision Tree implementation.
- `wilcoxon signed-rank test/` - Python implementatiion of Wilcoxon signed-Rank Test for perfromance on GP and MLP

##  Models

### 1. Genetic Programming (GP)
- Language: Java
- Be on this Directory: `cd gp_model`
- Run The Jar: `java -jar gp_forecaster.jar`

### 2. Multi-Layer Perceptron (MLP)
- Language: Java
- Dependencies: Listed in `mlp_model/requirements.txt`
- Be in this Directory: `cd mlp_model`
- Run The Jar: `java -jar mlp.jar`
`

### 3. Decision Tree (J48)
- Language: Java
- Be in this Directory: `cd decision_tree`
- command: `usr/lib/jvm/java-8-openjdk-amd64/bin/javac -cp lib/weka-3-8-0-monolithic.jar -d bin src/Decision_Tree_Classifier.java`
- Run The Jar: `java -Djava.awt.headless=true -cp bin:lib/weka-3-8-0-monolithic.jar src.Decision_Tree_Classifier`
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

2. Run the respective model scripts as described above.

## Group Mambers
  - Tiyiso Hlungwani : Genetic Programming
  - Junior Motsepe: Multi-Layer Perceptron
  - Tshegofatso Mapheto: Decision Tree
  - Tumisho Makhene: Report Writer
