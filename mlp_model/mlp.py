import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

#Load the data
train_df = pd.read_csv('../data/BTC_train.csv')
test_df = pd.read_csv('../data/BTC_test.csv')

#Preprocess Data
#Assuming last column is the label
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

#Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Build MLP Model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', random_state=42, max_iter=500)
mlp.fit(X_train_scaled, y_train)

#Predict and Evaluate
y_pred = mlp.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#Results
print("=== MLP Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
