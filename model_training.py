from sklearn.svm import SVC


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_svm(X_train, y_train, params):
    svm_grid = GridSearchCV(SVC(probability=True, random_state=42), params, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    return svm_grid.best_estimator_

def train_stacking(base_learners, meta_model, X_train, y_train):
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_model.fit(X_train, y_train)
    return stacking_model
