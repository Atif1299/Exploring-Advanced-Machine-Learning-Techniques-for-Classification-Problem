import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=1),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=1),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=1),
        "roc_auc": roc_auc_score(y_test, y_prob) if hasattr(model, "predict_proba") else 0,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics

def plot_model_comparison(results):
    results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.show()