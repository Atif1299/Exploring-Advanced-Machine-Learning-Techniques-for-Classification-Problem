from imblearn.over_sampling import SMOTE

def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)