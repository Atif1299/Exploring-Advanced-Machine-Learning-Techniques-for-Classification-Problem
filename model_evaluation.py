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

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
        plt.title(f"Feature Importance for {model_name}")
        plt.show()

def plot_confusion_matrix(model, X_test, y_test, model_name):
    cm = evaluate_model(model, X_test, y_test)["confusion_matrix"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
