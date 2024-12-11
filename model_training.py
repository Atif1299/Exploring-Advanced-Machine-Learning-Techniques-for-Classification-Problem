from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_random_forest(X_train, y_train, params):
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    return rf_grid.best_estimator_

def train_xgboost(X_train, y_train, params):
    xgb_grid = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0), params, cv=5, scoring='roc_auc', verbose=1, n_iter=10, random_state=42, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    return xgb_grid.best_estimator_
