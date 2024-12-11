from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

def train_svm(X_train, y_train, params):
    svm_grid = GridSearchCV(SVC(probability=True, random_state=42), params, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    return svm_grid.best_estimator_

def train_stacking(base_learners, meta_model, X_train, y_train):
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_model.fit(X_train, y_train)
    return stacking_model

def train_random_forest(X_train, y_train, params):
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    return rf_grid.best_estimator_

def train_xgboost(X_train, y_train, params):
    xgb_grid = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0), params, cv=5, scoring='roc_auc', verbose=1, n_iter=10, random_state=42, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    return xgb_grid.best_estimator_
