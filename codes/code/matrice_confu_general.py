

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def find_col(df, candidates):
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Fichier d'entrée (Excel coloré)")
    parser.add_argument("--true_col", type=str, default="Emotion", help="Colonne true")
    parser.add_argument("--pred_col", type=str, default="predicted_emotion", help="Colonne prédite")
    parser.add_argument("--labels", nargs="+", required=True, help="Liste des labels")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.input)

    true_col = find_col(df, [args.true_col, "Emotion", "Label", "true_emotion"])
    pred_col = find_col(df, [args.pred_col, "predicted_emotion", "Prediction", "predicted_emotion_final"])
    if true_col is None or pred_col is None:
        raise KeyError(f"[ERREUR] Colonnes introuvables dans {df.columns.tolist()}")

    before = len(df)
    df = df.dropna(subset=[true_col, pred_col])
    print(f"[INFO] {before - len(df)} lignes supprimées (NaN dans {true_col}/{pred_col})")

    y_true = df[true_col].astype(str).str.lower()
    y_pred = df[pred_col].astype(str).str.lower()
    

    report = classification_report(y_true, y_pred, output_dict=True)
    print("Macro-F1:", report["macro avg"]["f1-score"])
    print("Weighted-F1:", report["weighted avg"]["f1-score"])

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    class_metrics = pd.DataFrame(report).transpose()
    metrics_to_display = class_metrics.drop(["accuracy"], errors="ignore")
    axes[0].axis("off")
    table = axes[0].table(
        cellText=metrics_to_display.round(3).reset_index().values,
        colLabels=["Classe"] + metrics_to_display.columns.tolist(),
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    axes[0].set_title("Rapport de classification", fontsize=14)

    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[1])
    axes[1].set_title("Matrice de confusion", fontsize=14)
    axes[1].set_xlabel("Predicted Emotion")
    axes[1].set_ylabel("True Emotion")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/confusion_matrix_and_metrics.png", dpi=300)
    plt.close()

    class_metrics = pd.DataFrame(report).transpose().drop(["accuracy", "macro avg", "weighted avg"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_metrics.index, y=class_metrics["f1-score"], palette="viridis")
    plt.title("Per-class F1-score")
    plt.ylabel("F1-score")
    plt.xlabel("Emotion Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/per_class_f1.png", dpi=300)
    plt.close()

    minority_classes = [c for c in labels if c not in ["neutral", "joy"]]
    minority_errors = df[df[true_col].isin(minority_classes)]
    minority_errors = minority_errors[minority_errors[true_col] != minority_errors[pred_col]]
    misclass_counts = minority_errors[pred_col].value_counts(normalize=True) * 100

    if not misclass_counts.empty:
        plt.figure(figsize=(8, 8))
        plt.pie(
            misclass_counts,
            labels=misclass_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel")
        )
        plt.title("Misclassification Distribution of Minority Classes")
        plt.savefig(f"{args.output_dir}/minority_errors_pie.png", dpi=300)
        plt.close()

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{args.output_dir}/classification_report.csv", index=True)

    print(f"[OK] Résultats sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
