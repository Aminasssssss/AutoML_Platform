# AutoML Platform

![Python](https://img.shields.io/badge/python-3.11-FFB6C1?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.31-FF69B4?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-2.10-FF69B4?style=flat-square&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-FFC0CB?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-FFB6C1?style=flat-square)

Automated machine learning platform — upload any CSV dataset and get full EDA, model training, 5-fold cross-validation, SHAP explanations, model download, and a PDF report in one click.

## Supported Tasks

This platform supports **classification tasks** — binary and multiclass.

Works with:
- Churn prediction (yes/no)
- Medical diagnosis (sick/healthy)
- Fraud detection (fraud/not fraud)
- Customer segmentation (3+ classes)
- Any CSV with a categorical target column

Does not support:
- Regression (predicting a number, e.g. price or salary)
- Time series forecasting
- Text or image data

## What it does

- Automated EDA with visualizations (distributions, correlations, missing values, class balance)
- Data preprocessing — missing values, encoding, scaling, SMOTE, datetime column detection
- Trains 6 ML models simultaneously: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- 5-fold cross-validation for reliable metrics
- MLflow experiment tracking for all runs
- SHAP explanations (TreeExplainer → LinearExplainer → KernelExplainer fallback)
- Download best model as .pkl
- One-click PDF report with EDA + model comparison + SHAP
- Built-in sample dataset (Titanic) to try without uploading

## Results on sample datasets

| Dataset | Best Model | ROC-AUC | CV ROC-AUC | F1 |
|---------|-----------|---------|------------|-----|
| Telco Churn | LightGBM | 0.847 | 0.841 ± 0.012 | 0.612 |
| Breast Cancer | Random Forest | 0.997 | 0.995 ± 0.003 | 0.971 |
| Titanic | XGBoost | 0.876 | 0.869 ± 0.018 | 0.802 |

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open http://localhost:8501, upload any CSV or use the built-in Titanic sample, select target column, click Run AutoML.

## Docker

```bash
docker build -t automl-platform .
docker run -p 8501:8501 automl-platform
```

## MLflow tracking

```bash
mlflow ui
```

Open http://localhost:5000 to see all experiments and model runs.

## Project Structure

```
automl-platform/
├── src/
│   ├── eda.py           # EDA analysis and visualizations
│   ├── preprocessor.py  # Data cleaning, encoding, scaling, datetime handling
│   ├── trainer.py       # 6 models + 5-fold CV + MLflow + model saving
│   ├── explainer.py     # SHAP with 3-level fallback
│   └── pdf_report.py    # PDF report generation
├── app/
│   └── streamlit_app.py # Streamlit UI with progress bar
├── models/              # Saved model files (.pkl)
├── reports/             # Generated PDF reports
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
└── requirements.txt
```

## Tech Stack

Python | scikit-learn | XGBoost | LightGBM | CatBoost | SHAP | MLflow | Streamlit | Matplotlib | Seaborn | Docker

## Author

Zhumatayeva Amina — 2nd year Information Systems, KBTU