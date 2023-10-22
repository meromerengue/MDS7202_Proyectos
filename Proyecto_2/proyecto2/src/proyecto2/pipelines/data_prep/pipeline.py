"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess,
    get_data
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs='dataset',
                name="nodo_data",
            ),
            node(
                func=preprocess,
                inputs="dataset",
                outputs="dataset_preproc",
                name="nodo_preproc",
            )
        ]
    )

