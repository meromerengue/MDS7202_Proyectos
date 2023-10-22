"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, optimize


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["dataset_preproc", "params:split_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="nodo_split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="best_model",
                name="nodo_entrenamiento",
            ),
            node(
                func=evaluate_model,
                inputs=["best_model", "X_test", "y_test"],
                outputs=None,
                name="nodo_eval",
            ),
            node(
                func=optimize,
                inputs=["best_model", 'X_train', 'X_test', 'y_train', 'y_test'],
                outputs='best_opti_model',
                name='nodo_opti'
            )
        ]
    )
