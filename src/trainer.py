import numpy as np
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                              precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))

    if n_classes == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
    else:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(label_binarize(y_test, classes=np.unique(y_test)),
                                 y_proba, multi_class='ovr')
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def cross_validate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    return round(scores.mean(), 4), round(scores.std(), 4)


def train_all(X, y, experiment_name="automl-platform"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        classes, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        if min_count >= 6:
            k_neighbors = min(5, min_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except Exception:
        X_train_res, y_train_res = X_train, y_train

    mlflow.set_experiment(experiment_name)
    models = get_models()
    results = {}

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train_res, y_train_res)
            metrics = evaluate(model, X_test, y_test)

            cv_mean, cv_std = cross_validate_model(model, X, y)
            metrics['cv_roc_auc_mean'] = cv_mean
            metrics['cv_roc_auc_std'] = cv_std

            mlflow.log_params({"model": name, "cv_folds": 5})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            safe_name = name.replace(" ", "_").lower()
            model_path = f"models/{safe_name}.pkl"
            joblib.dump(model, model_path)

            results[name] = {
                "model": model,
                "metrics": metrics,
                "model_path": model_path
            }

    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])

    joblib.dump(results[best_name]["model"], "models/best_model.pkl")

    return results, best_name, X_test, y_test