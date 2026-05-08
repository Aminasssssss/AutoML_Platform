import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eda import run_eda, generate_eda_figures
from src.preprocessor import preprocess
from src.trainer import train_all
from src.explainer import plot_shap_bar, plot_shap_summary
from src.pdf_report import generate_pdf_report

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #FF69B4; }
    .section-title { font-size: 1.2rem; font-weight: 600; color: #FF69B4; margin-top: 1rem; }
    .stButton>button { background-color: #FF69B4; color: white; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #FF1493; }
    .stDownloadButton>button { background-color: #FF69B4; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AutoML Platform</p>', unsafe_allow_html=True)
st.markdown("Upload any CSV dataset and get automated EDA, model training, SHAP explanations, and a PDF report.")

st.sidebar.title("Settings")

# Sample dataset option
use_sample = st.sidebar.checkbox("Use sample dataset (Titanic)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload your CSV", type=["csv"])

if use_sample:
    import urllib.request
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
        st.sidebar.success("Sample dataset loaded: Titanic (891 rows)")
        default_target = 'Survived'
    except Exception:
        st.sidebar.error("Could not load sample dataset")
        df = None
        default_target = None
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    default_target = df.columns[-1]
else:
    df = None
    default_target = None

if df is not None:
    target_col = st.sidebar.selectbox(
        "Select target column",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_target) if default_target in df.columns else 0
    )

    st.dataframe(df.head(5), use_container_width=True)

    if st.sidebar.button("Run AutoML"):

        progress = st.progress(0)
        status = st.empty()

        status.text("Running EDA...")
        progress.progress(10)
        eda_report = run_eda(df, target_col)
        eda_figures = generate_eda_figures(df, target_col)

        st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", eda_report['shape'][0])
        col2.metric("Columns", eda_report['shape'][1])
        col3.metric("Missing Values", sum(v for v in eda_report['missing'].values() if v > 0))
        col4.metric("Duplicates", eda_report['duplicates'])

        st.markdown('<p class="section-title">EDA Visualizations</p>', unsafe_allow_html=True)
        for title, fig in eda_figures:
            st.subheader(title)
            st.pyplot(fig)
            plt.close(fig)

        progress.progress(30)
        status.text("Preprocessing data...")
        X, y, feature_names = preprocess(df, target_col)

        progress.progress(40)
        status.text("Training 6 models with 5-fold cross-validation...")
        results, best_name, X_test, y_test = train_all(X, y)

        progress.progress(80)

        st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)

        metrics_data = []
        for name, res in results.items():
            m = res['metrics']
            metrics_data.append({
                'Model': name,
                'Accuracy': m['accuracy'],
                'F1': m['f1'],
                'ROC-AUC': m['roc_auc'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'CV ROC-AUC': f"{m.get('cv_roc_auc_mean', '—')} ± {m.get('cv_roc_auc_std', '')}" if m.get('cv_roc_auc_mean') else '—'
            })

        metrics_df = pd.DataFrame(metrics_data).sort_values('ROC-AUC', ascending=False)

        def highlight_best(row):
            return ['background-color: #fff0f5; font-weight: bold'
                    if row['Model'] == best_name else '' for _ in row]

        st.dataframe(metrics_df.style.apply(highlight_best, axis=1), use_container_width=True)
        st.success(f"Best model: {best_name} — ROC-AUC: {results[best_name]['metrics']['roc_auc']} | CV: {results[best_name]['metrics'].get('cv_roc_auc_mean', 'N/A')}")

        # Download best model
        st.markdown('<p class="section-title">Download Best Model</p>', unsafe_allow_html=True)
        if os.path.exists("models/best_model.pkl"):
            with open("models/best_model.pkl", "rb") as f:
                st.download_button(
                    label=f"Download {best_name} (.pkl)",
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )

        progress.progress(85)
        status.text("Computing SHAP explanations...")

        st.markdown('<p class="section-title">SHAP Explanations</p>', unsafe_allow_html=True)
        best_model = results[best_name]['model']
        shap_bar_fig = plot_shap_bar(best_model, X_test, feature_names)
        shap_summary_fig = plot_shap_summary(best_model, X_test, feature_names)
        st.pyplot(shap_bar_fig)
        st.pyplot(shap_summary_fig)

        progress.progress(90)
        status.text("Generating PDF report...")

        st.markdown('<p class="section-title">Download PDF Report</p>', unsafe_allow_html=True)
        os.makedirs("reports", exist_ok=True)
        report_path = "reports/automl_report.pdf"

        generate_pdf_report(
            eda_report=eda_report,
            eda_figures=generate_eda_figures(df, target_col),
            results=results,
            best_name=best_name,
            shap_fig=shap_bar_fig,
            feature_names=feature_names,
            output_path=report_path
        )

        with open(report_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="automl_report.pdf",
                mime="application/pdf"
            )

        progress.progress(100)
        status.text("Done!")

else:
    st.info("Upload a CSV file from the sidebar or use the sample Titanic dataset.")
    st.markdown("""
    **What this platform does:**
    - Automated EDA with visualizations
    - Data preprocessing (missing values, encoding, scaling, SMOTE)
    - Trains 6 ML models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
    - 5-fold cross-validation for reliable metrics
    - Compares all models by Accuracy, F1, ROC-AUC, Precision, Recall
    - SHAP explanations for the best model
    - Download best model as .pkl
    - One-click PDF report with everything included

    **Supported tasks:** Binary and multiclass classification
    """)