import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

# Prompts
try:
    seed = int(input("Enter seed value (integer): "))
except ValueError:
    print("Invalid seed value. Must be an integer.")
    sys.exit(1)

train_path = input("Enter path to training CSV: ").strip()
test_path = input("Enter path to test CSV: ").strip()

# Load the data
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
except FileNotFoundError:
    print("Error: One or both CSV file paths are incorrect.")
    sys.exit(1)

if train_df.empty or test_df.empty:
    print("Error: One of the CSV files is empty.")
    sys.exit(1)

# Preprocess Data (last column assumed to be label)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train MLP model with provided seed
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    solver='adam',
    random_state=seed,
    max_iter=500
)
mlp.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = mlp.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print("=== MLP Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
