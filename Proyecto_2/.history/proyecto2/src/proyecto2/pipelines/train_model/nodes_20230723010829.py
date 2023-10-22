"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

import logging
from typing import Dict

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score
from sklearn.svm import SVR
from xgboost import XGBRegressor


from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna
import mlflow




def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]

    train_idx = int(rows * train_ratio)

    assert rows > train_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_test, y_test = X[train_idx:], y[train_idx:]

    return X_train, X_test, y_train, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.test_recall")["run_id"].iloc[-1]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    print(type(best_model).__name__)
    return best_model


# TODO: completar train_model
def train_model(X_train, X_test, y_train, y_test):
    modelos = [
    ('DummyClassifier', DummyClassifier(strategy='stratified', random_state=23)),
    ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=23)),
    #('KNeighborsClassifier', KNeighborsClassifier()),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=23)),
    ('SVC', SVC(random_state=23)),
    ('RandomForestClassifier', RandomForestClassifier(random_state=23)),
    ('LightGBMClassifier', LGBMClassifier(random_state=23)),
    ('XGBClassifier', XGBClassifier())
    ]
    mlflow.create_experiment("proyecto2_baseline_mlflow")
    experiment = mlflow.get_experiment_by_name("proyecto2_baseline_mlflow")
    experiment_id = experiment.experiment_id
    for model_name, model_instance in modelos:
        mlflow.autolog()
        with mlflow.start_run(
            run_name=model_name + "_run", experiment_id=experiment_id
        ):
            model_instance.fit(X_train, y_train)
            predictions = model_instance.predict(X_test)
            test_recall = recall_score(y_test, predictions)
            mlflow.log_metric("test_recall", test_recall)
    return get_best_model(experiment_id)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Recall of {recall} on test data.")


def optim(model, X_train, X_test, y_train, y_test):
    tipo_modelo = type(model).__name__

    def objective(trial):
        hyperparameters = {
        "Dummy": {
            "strategy": trial.suggest_categorical("strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]),
        },
        "LogisticRegression": {
            "C": trial.suggest_loguniform("C", 1e-6, 1e6),
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
        },
        "DecisionTreeClassifier": {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        },
        "SVC": {
            "C": trial.suggest_loguniform("C", 1e-6, 1e6),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])#,
            #"gamma": trial.suggest_categorical("gamma", ["scale", "auto"]) if trial.suggest_categorical("kernel", ["poly", "rbf", "sigmoid"]) else None,
        },
        "RandomForestClassifier": {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        },
        "LightGBMClassifier": {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        },
        "XGBClassifier": {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
        },
        }
        grilla = hyperparameters[tipo_modelo]
        print(grilla)
        modelfit = model.set_params(**grilla)
        print(type(modelfit).__name__)
        modelfit.fit(X_train, y_train)
        y_pred = modelfit.predict(X_test)
        print(y_pred)

        return recall_score(y_pred, y_test)
    
    mlflow.create_experiment("proyecto2_opti_mlflow")
    experiment = mlflow.get_experiment_by_name("proyecto2_opti_mlflow")
    experiment_id = experiment.experiment_id
    mlflow.autolog()
    with mlflow.start_run(
            run_name="opti_run", experiment_id=experiment_id
        ):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Realizar la búsqueda de hiperparámetros con Optuna
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(objective, n_trials=1)

        best_params_optuna = study_rf.best_params
        best_score = study_rf.best_value
        mlflow.log_metric("test_recall", best_score)
    print(best_score)    
    print(best_params_optuna)    
    return model.set_params(**best_params_optuna)



