"""
This is a boilerplate pipeline 'shap_importances'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_shap_values


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_shap_values,
                inputs=['best_opti_model', 'dataset_preproc'],
                outputs='shap_values',
                name="nodo_shap",
            )    
    ]
    )
