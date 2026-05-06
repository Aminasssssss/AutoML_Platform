import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')


def get_shap_values(model, X_test: pd.DataFrame):
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test[:100])
        return explainer, shap_values
    except Exception:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test[:100])
            return explainer, shap_values
        except Exception:
            return None, None


def plot_shap_summary(model, X_test: pd.DataFrame, feature_names: list):
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test[:100])
        shap.plots.beeswarm(shap_values, show=False)
        fig = plt.gcf()
    except Exception:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test[:100])
            if isinstance(sv, list):
                sv = sv[1]
            shap.summary_plot(sv, X_test[:100], feature_names=feature_names, show=False)
            fig = plt.gcf()
        except Exception:
            ax.text(0.5, 0.5, 'SHAP not available for this model',
                    ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    return fig


def plot_shap_bar(model, X_test: pd.DataFrame, feature_names: list):
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test[:100])
        vals = np.abs(shap_values.values).mean(0)
        if vals.ndim > 1:
            vals = vals.mean(1)
        feat_imp = pd.Series(vals, index=feature_names).sort_values(ascending=True).tail(15)
        feat_imp.plot(kind='barh', ax=ax, color='#FF69B4')
        ax.set_title('SHAP Feature Importance', fontweight='bold')
        ax.set_xlabel('Mean |SHAP value|')
    except Exception:
        ax.text(0.5, 0.5, 'SHAP not available for this model',
                ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    return fig
