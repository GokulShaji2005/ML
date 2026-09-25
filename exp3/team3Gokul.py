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


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

data = load_breast_cancer()

X = data.data

# Original sklearn dataset:
# 0 = Malignant
# 1 = Benign
#
# We reverse it so that:
# 0 = Benign
# 1 = Malignant

y = 1 - data.target

print("Dataset Shape:", X.shape)

print("\nClass Distribution:")
print(
    pd.Series(y).value_counts().sort_index()
)

print("\nClass Names:")
print("0 = Benign")
print("1 = Malignant")


# ==========================================================
# 2. TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================================
# 3. FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================================
# 4. TRAIN LOGISTIC REGRESSION
# ==========================================================
#
# L2 regularization with C=0.1
#
# Smaller C = stronger regularization
# This helps avoid extremely confident 0/1 probabilities.
#

model = LogisticRegression(
    penalty="l2",
    C=0.1,
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully.")


# ==========================================================
# 5. GET PREDICTED PROBABILITIES
# ==========================================================

# Probability of class 1 = Malignant

y_prob = model.predict_proba(X_test)[:, 1]

print("\nFirst 10 Predicted Probabilities:")
print(y_prob[:10])

print("\nProbability Statistics:")
print("Minimum:", y_prob.min())
print("Maximum:", y_prob.max())
print("Mean:", y_prob.mean())


# ==========================================================
# 6. THRESHOLDS TO INVESTIGATE
# ==========================================================

thresholds = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]


# ==========================================================
# 7. EVALUATE EACH THRESHOLD
# ==========================================================

results = []

for threshold in thresholds:

    # ------------------------------------------------------
    # Apply threshold
    # ------------------------------------------------------

    y_pred = (y_prob >= threshold).astype(int)


    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()


    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ------------------------------------------------------
    # Store results
    # ------------------------------------------------------

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "False Positives": fp,
        "False Negatives": fn
    })


# ==========================================================
# 8. CREATE COMPARISON TABLE
# ==========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 110)
print("THRESHOLD VS METRICS")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format
        }
    )
)


# ==========================================================
# 9. SAVE TABLE
# ==========================================================

results_df.to_csv(
    "threshold_results.csv",
    index=False
)

print("\nResults saved to: threshold_results.csv")


# ==========================================================
# 10. CONFUSION MATRICES
# ==========================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nThreshold:", threshold)

    print(
        "              Predicted"
    )

    print(
        "              Benign  Malignant"
    )

    print(
        f"Actual Benign    {cm[0,0]:3d}      {cm[0,1]:3d}"
    )

    print(
        f"Actual Malignant {cm[1,0]:3d}      {cm[1,1]:3d}"
    )


# ==========================================================
# 11. PLOT PRECISION, RECALL AND F1
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    linewidth=2,
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1-Score"],
    marker="o",
    linewidth=2,
    label="F1-Score"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Effect of Classification Threshold on Precision, Recall and F1-Score"
)

plt.xticks(thresholds)

plt.ylim(0, 1.05)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "threshold_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Metrics graph saved to: threshold_metrics.png"
)


# ==========================================================
# 12. PLOT FALSE POSITIVES AND FALSE NEGATIVES
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["False Positives"],
    marker="o",
    linewidth=2,
    label="False Positives"
)

plt.plot(
    results_df["Threshold"],
    results_df["False Negatives"],
    marker="o",
    linewidth=2,
    label="False Negatives"
)

plt.xlabel("Classification Threshold")

plt.ylabel("Number of Cases")

plt.title(
    "Effect of Classification Threshold on False Positives and False Negatives"
)

plt.xticks(thresholds)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "threshold_fp_fn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "FP/FN graph saved to: threshold_fp_fn.png"
)


# ==========================================================
# 13. FINISHED
# ==========================================================

print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print("1. threshold_results.csv")
print("2. threshold_metrics.png")
print("3. threshold_fp_fn.png")
