from data_preprocessing import load_and_preprocess_data, encode_categorical_columns
from visualization import visualize_columns
from model_training import train_random_forest, train_svm, train_xgboost, train_stacking
from model_evaluation import evaluate_model, plot_model_comparison, plot_confusion_matrix, plot_feature_importance
from utils import handle_imbalance, scale_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

def main():
    file_path = "Weather_Data.csv"
    data = load_and_preprocess_data(file_path)
    
    categorical_columns = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"]
    data = encode_categorical_columns(data, categorical_columns)

    selected_features = [
        "MinTemp", "MaxTemp", "Rainfall", "Humidity3pm",
        "Pressure9am", "WindGustSpeed", "Temp9am", "Temp3pm",
        "Cloud9am", "Cloud3pm"
    ]
    X = data[selected_features]
    y = data["RainTomorrow"]

    visualize_columns(data, selected_features)

    # Handle imbalanced data and split
    X_resampled, y_resampled = handle_imbalance(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

    feature_names = X_train.columns

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    svm_params = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    best_rf = train_random_forest(X_train_scaled, y_train, rf_params)
    best_svm = train_svm(X_train_scaled, y_train, svm_params)
    best_xgb = train_xgboost(X_train_scaled, y_train, xgb_params)

    base_learners = [
        ('random_forest', best_rf),
        ('svm', best_svm),
        ('xgboost', best_xgb)
    ]
    meta_model = LogisticRegression(random_state=42)
    stacking_model = train_stacking(base_learners, meta_model, X_train_scaled, y_train)

    # Evaluate models
    models = {
        "Random Forest": best_rf,
        "SVM": best_svm,
        "XGBoost": best_xgb,
        "Stacking Classifier": stacking_model
    }

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test_scaled, y_test)
        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"]
        })
        plot_confusion_matrix(model, X_test_scaled, y_test, name)
        
        # Call plot_feature_importance for models with feature_importances_
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, feature_names, name)

    # Convert results to DataFrame and plot model comparison
    results_df = pd.DataFrame(results)
    plot_model_comparison(results_df)

if __name__ == "__main__":
    main()
