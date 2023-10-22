"""
This is a boilerplate pipeline 'shap_importances'
generated using Kedro 0.18.11
"""
import shap
import pandas as pd

def get_shap_values(model, X):
    X = X.drop(columns='credit_score')
    X_shap = model.fit_transform(X)
    feature_names = X_shap.columns
    # Calcular los valores SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_values_df = pd.DataFrame(shap_values[0][:][:], columns = feature_names)
    return shap_values_df
