import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_eda(df: pd.DataFrame, target_col: str) -> dict:
    report = {}

    report['shape'] = df.shape
    report['dtypes'] = df.dtypes.astype(str).to_dict()
    report['missing'] = df.isnull().sum().to_dict()
    report['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    report['duplicates'] = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    report['numeric_cols'] = numeric_cols
    report['categorical_cols'] = df.select_dtypes(include='object').columns.tolist()
    report['target_col'] = target_col
    report['target_distribution'] = df[target_col].value_counts().to_dict()
    report['class_balance'] = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()

    if numeric_cols:
        report['stats'] = df[numeric_cols].describe().round(4).to_dict()
        report['correlations'] = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col).round(4).to_dict()

    return report


def generate_eda_figures(df: pd.DataFrame, target_col: str) -> list:
    figures = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    sns.set_theme(style='whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], color='#FF69B4', edgecolor='white')
    axes[0].set_title('Target Distribution', fontweight='bold')
    axes[0].tick_params(rotation=0)
    df[target_col].value_counts(normalize=True).plot(kind='pie', ax=axes[1],
        autopct='%1.1f%%', colors=['#FFB6C1', '#FF69B4', '#FFC0CB', '#DB7093'])
    axes[1].set_title('Target Balance', fontweight='bold')
    axes[1].set_ylabel('')
    plt.tight_layout()
    figures.append(('Target Distribution', fig))

    if numeric_cols:
        n = len(numeric_cols)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, col in enumerate(numeric_cols[:9]):
            df[col].hist(ax=axes[i], bins=30, color='#FFB6C1', edgecolor='white')
            axes[i].set_title(col, fontweight='bold')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        figures.append(('Feature Distributions', fig))

        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols + [target_col]].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdPu', center=0, ax=ax, linewidths=0.5)
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            figures.append(('Correlation Heatmap', fig))

    if numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        corr_with_target = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col).sort_values()
        colors = ['#FF69B4' if v > 0 else '#FFB6C1' for v in corr_with_target]
        corr_with_target.plot(kind='barh', ax=ax, color=colors)
        ax.set_title('Feature Correlation with Target', fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        figures.append(('Feature Correlations with Target', fig))

    return figures
