import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
data = load_breast_cancer()

x = data.data
y = data.target

print("Dataset Shape:", x.shape)
print("Classes:", np.unique(y))

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create Logistic Regression model
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

# Train model
model.fit(x_train, y_train)

# Get probability of class 1
y_prob = model.predict_proba(x_test)[:, 1]

print("\nFirst 10 predicted probabilities:")
print(y_prob[:10])


# Function to evaluate different thresholds
def evaluate_threshold(threshold):

    # Convert probabilities into class predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract values
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 45)
    print("Threshold:", threshold)
    print("=" * 45)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nFalse Positives:", fp)
    print("False Negatives:", fn)

    return accuracy, precision, recall, f1, fp, fn


# Test different thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]

results = []

for threshold in thresholds:

    accuracy, precision, recall, f1, fp, fn = evaluate_threshold(threshold)

    results.append([
        threshold,
        accuracy,
        precision,
        recall,
        f1,
        fp,
        fn
    ])


# Create results table
results_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Positives",
        "False Negatives"
    ]
)

print("\n\nRESULTS TABLE")
print(results_df)


# Plot Precision and Recall
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker='o',
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker='o',
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1 Score"],
    marker='o',
    label="F1 Score"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Effect of Decision Threshold on Model Performance")
plt.legend()
plt.grid(True)

plt.show()


# Plot False Positives and False Negatives
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["False Positives"],
    marker='o',
    label="False Positives"
)

plt.plot(
    results_df["Threshold"],
    results_df["False Negatives"],
    marker='o',
    label="False Negatives"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Number of Cases")
plt.title("Effect of Threshold on False Positives and False Negatives")
plt.legend()
plt.grid(True)

plt.show()
