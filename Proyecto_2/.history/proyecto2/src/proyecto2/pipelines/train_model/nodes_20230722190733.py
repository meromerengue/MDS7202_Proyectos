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
    best_model_id = runs.sort_values("metrics.test_recall")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

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
    mlflow.create_experiment("proyecto2_mlflow")
    experiment = mlflow.get_experiment_by_name("proyecto2_mlflow")
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
