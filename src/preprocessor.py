import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df = df.drop_duplicates()

    # Drop datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
    for col in df.columns:
        if col != target_col:
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                if df[col].dtype == object:
                    sample = df[col].dropna().head(10)
                    if all('-' in str(v) or '/' in str(v) for v in sample):
                        datetime_cols.append(col)
            except Exception:
                pass
    if datetime_cols:
        df = df.drop(columns=[c for c in datetime_cols if c != target_col])

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, list(X.columns)