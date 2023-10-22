"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["companies", "shuttles", "reviews"],
                name="nodo_data",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="companies_preprocess",
                name="nodo_precom",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="shuttles_preprocess",
                name="nodo_preshu",
            ),
            node(
                func=create_model_input_table,
                inputs=["shuttles_preprocess", "companies_preprocess", "reviews"],
                outputs="df_model",
                name="nodo_prefin",
            ),
        ]
    )
