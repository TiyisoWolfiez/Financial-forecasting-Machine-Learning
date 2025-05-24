# Decision Tree (J48) Classifier

A Java implementation of the J48 decision tree classifier using the Weka machine learning library.

## Requirements

- **Language:** Java 8+
- **Dependencies:** Weka 3.8 (weka-3-8-0-monolithic.jar included in `lib/`)

## Setup

Navigate to the project directory:
```bash
cd decision_tree
```

## Building the JAR

Compile the Java source code and create the JAR file:

```bash
/usr/lib/jvm/java-8-openjdk-amd64/bin/javac -cp lib/weka-3-8-0-monolithic.jar -d bin src/Decision_Tree_Classifier.java
cd bin
jar cf ../DecisionTree.jar src/*.class
cd ..
```

## Running the Application

Execute the compiled JAR file:

```bash
/usr/lib/jvm/java-8-openjdk-amd64/bin/java -Djava.awt.headless=true -cp "DecisionTree.jar:lib/weka-3-8-0-monolithic.jar" src.Decision_Tree_Classifier
```

## Usage

When you run the application, you will be prompted to provide:

1. **Training CSV file path** - Path to your training dataset
2. **Testing CSV file path** - Path to your testing dataset  
3. **Random seed value** - Integer value for reproducible results

## Output

The classification results will be automatically saved to `DecisionTree_Results.csv` in the current directory.

## Quick Start

If you prefer to skip the compilation step, you can use the prebuilt `DecisionTree.jar` included in this repository. Just ensure that Java 8+ and the Weka JAR file are available in your environment.

## Directory Structure

```
decision_tree/
├── src/
│   └── Decision_Tree_Classifier.java
├── lib/
│   └── weka-3-8-0-monolithic.jar
├── bin/
│   └── (compiled classes)
├── DecisionTree.jar
└── DecisionTree_Results.csv (generated after running)
```