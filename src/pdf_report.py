import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from datetime import datetime


def generate_pdf_report(
    eda_report: dict,
    eda_figures: list,
    results: dict,
    best_name: str,
    shap_fig,
    feature_names: list,
    output_path: str = "reports/automl_report.pdf"
):
    with pdf_backend.PdfPages(output_path) as pdf:

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        title_text = "AutoML Platform — Automated Analysis Report"
        subtitle = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ax.text(0.5, 0.7, title_text, ha='center', va='center',
                fontsize=20, fontweight='bold', color='#FF69B4', transform=ax.transAxes)
        ax.text(0.5, 0.4, subtitle, ha='center', va='center',
                fontsize=12, color='gray', transform=ax.transAxes)
        ax.text(0.5, 0.2, f"Target: {eda_report.get('target_col', 'unknown')} | "
                           f"Dataset shape: {eda_report.get('shape', 'N/A')} | "
                           f"Best model: {best_name}",
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        info = [
            ["Metric", "Value"],
            ["Dataset shape", str(eda_report.get('shape', 'N/A'))],
            ["Duplicates removed", str(eda_report.get('duplicates', 0))],
            ["Numeric features", str(len(eda_report.get('numeric_cols', [])))],
            ["Categorical features", str(len(eda_report.get('categorical_cols', [])))],
            ["Target column", str(eda_report.get('target_col', 'N/A'))],
        ]
        missing = eda_report.get('missing', {})
        missing_total = sum(v for v in missing.values() if v > 0)
        info.append(["Total missing values", str(missing_total)])

        table = ax.table(cellText=info[1:], colLabels=info[0],
                         cellLoc='left', loc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.5, 2)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#FFB6C1')
                cell.set_text_props(fontweight='bold')
        ax.set_title('Dataset Overview', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for title, fig in eda_figures:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        metrics_data = []
        for name, res in results.items():
            m = res['metrics']
            metrics_data.append([
                name,
                str(m['accuracy']),
                str(m['f1']),
                str(m['roc_auc']),
                str(m['precision']),
                str(m['recall']),
            ])

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('off')
        headers = ['Model', 'Accuracy', 'F1', 'ROC-AUC', 'Precision', 'Recall']
        table = ax.table(cellText=metrics_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#FFB6C1')
                cell.set_text_props(fontweight='bold')
            elif any(best_name in row[0] for row in [metrics_data[r-1]] if r > 0):
                cell.set_facecolor('#FFF0F5')
        ax.set_title('Model Comparison Results', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if shap_fig is not None:
            pdf.savefig(shap_fig, bbox_inches='tight')
            plt.close(shap_fig)

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('off')
        ax.text(0.5, 0.6, f"Best Model: {best_name}",
                ha='center', va='center', fontsize=16,
                fontweight='bold', color='#FF69B4', transform=ax.transAxes)
        best_metrics = results[best_name]['metrics']
        summary = (f"Accuracy: {best_metrics['accuracy']} | "
                   f"F1: {best_metrics['f1']} | "
                   f"ROC-AUC: {best_metrics['roc_auc']}")
        ax.text(0.5, 0.3, summary, ha='center', va='center',
                fontsize=13, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return output_path
