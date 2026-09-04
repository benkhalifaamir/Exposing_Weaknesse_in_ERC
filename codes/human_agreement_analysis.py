
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from dataset_config import get_config, SUPPORTED_DATASETS
from erc_common import load_reannotation


def average_pairwise_kappa(df, ann_cols):
    kappas = []
    for i in range(len(ann_cols)):
        for j in range(i + 1, len(ann_cols)):
            sub = df[[ann_cols[i], ann_cols[j]]].dropna()
            if len(sub) < 2:
                continue
            k = cohen_kappa_score(sub[ann_cols[i]], sub[ann_cols[j]])
            kappas.append(k)
    return float(np.mean(kappas)) if kappas else float("nan"), kappas


def agreement_regime(row, ann_cols):
    values = [row[c] for c in ann_cols if pd.notna(row[c])]
    n = len(values)
    if n == 0:
        return None, 0, n
    counts = pd.Series(values).value_counts()
    top_count = int(counts.iloc[0])
    top_label = counts.index[0]
    return f"{top_count}/{n}", top_count, n


def agreement_level_3class(top_count, n):
    ratio = top_count / n if n else 0
    if ratio <= 0.4:
        return "faible"
    elif ratio <= 0.6:
        return "moyen"
    else:
        return "eleve"


CUE_COLUMNS = ["Intent_Clarity", "Intent_Count", "Context_Clarity", "Lexical_Cue_Present", "Punctuation"]


def train_agreement_decision_tree(merged, output_dir, dataset):
    present_cues = [c for c in CUE_COLUMNS if c in merged.columns]
    if len(present_cues) < 2:
        print(f"  [INFO] Pas assez d'indices linguistiques presents ({present_cues}), arbre de decision ignore.")
        return None

    data = merged.dropna(subset=present_cues + ["agreement_level"]).copy()
    if len(data) < 10:
        print("  [INFO] Trop peu d'exemples avec indices + niveau d'accord pour entrainer l'arbre.")
        return None

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = encoder.fit_transform(data[present_cues].astype(str))
    y = data["agreement_level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(f"  Arbre de decision (accord humain <- indices linguistiques) sur {dataset} :")
    print(f"    Exactitude : {report['accuracy']:.2f}")
    print(f"    Precision (macro) : {report['macro avg']['precision']:.2f}")
    print(f"    Rappel (macro)    : {report['macro avg']['recall']:.2f}")
    print(f"    F1 (macro)        : {report['macro avg']['f1-score']:.2f}")

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(clf, feature_names=present_cues, class_names=sorted(y.unique()), filled=True, fontsize=8, ax=ax)
    fig.savefig(Path(output_dir) / f"{dataset}_decision_tree_accord.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyse de l'accord inter-annotateurs (chapitre 4)")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--input", required=True, help="Fichier de reannotation fusionne (csv/xlsx)")
    parser.add_argument("--cues", default=None, help="Fichier d'indices linguistiques (*_cues_annotations.xlsx)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "files" / args.dataset / "chapitre4_accord_humain"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Chargement de {args.input} ...")
    df, ann_cols = load_reannotation(args.input, args.dataset)
    print(f"  Colonnes d'annotation detectees ({len(ann_cols)}) : {ann_cols}")
    if len(ann_cols) < 5:
        print(f"  [ATTENTION] Le memoire decrit 4 annotateurs humains + etiquette originale (5 annotations). "
              f"Seules {len(ann_cols)} colonnes sont disponibles pour {args.dataset} -- voir README.")

    print("[2/4] Cohen's kappa moyen entre annotateurs...")
    mean_kappa, all_kappas = average_pairwise_kappa(df, ann_cols)
    print(f"  Cohen's kappa moyen ({args.dataset}) : {mean_kappa:.2f}  (sur {len(all_kappas)} paires)")

    print("[3/4] Regimes d'accord par enonce...")
    regimes, top_counts, ns = [], [], []
    for _, row in df.iterrows():
        regime, top_count, n = agreement_regime(row, ann_cols)
        regimes.append(regime)
        top_counts.append(top_count)
        ns.append(n)
    df["agreement_regime"] = regimes
    df["agreement_level"] = [agreement_level_3class(tc, n) for tc, n in zip(top_counts, ns)]

    print("  Distribution des regimes d'accord :")
    print(df["agreement_regime"].value_counts().sort_index().to_string())
    print("  Distribution faible/moyen/eleve :")
    print(df["agreement_level"].value_counts().to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    df["agreement_regime"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution des regimes d'accord - {args.dataset}")
    ax.set_xlabel("Regime d'accord")
    ax.set_ylabel("Nombre d'enonces")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.dataset}_agreement_distribution.png", dpi=200)
    plt.close(fig)

    merged = df
    if args.cues:
        print(f"[4/4] Fusion avec les indices linguistiques ({args.cues}) et entrainement de l'arbre de decision...")
        cues_df = pd.read_excel(args.cues) if str(args.cues).endswith(".xlsx") else pd.read_csv(args.cues)
        key_cols = [c for c in ["dialog_id", "speaker", "utterance"] if c in df.columns and c in cues_df.columns]
        if key_cols:
            merged = df.merge(cues_df[key_cols + [c for c in CUE_COLUMNS if c in cues_df.columns]], on=key_cols, how="left")
        else:
            print("  [WARN] Aucune cle commune trouvee pour la fusion, alignement par position (index).")
            for c in CUE_COLUMNS:
                if c in cues_df.columns:
                    merged[c] = cues_df[c].values[:len(merged)]
        train_agreement_decision_tree(merged, out_dir, args.dataset)
    else:
        print("[4/4] Pas de fichier --cues fourni : arbre de decision (Tableau 4.5) ignore.")

    out_csv = out_dir / f"{args.dataset}_agreement_analysis.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Resultats sauvegardes dans {out_dir} (dont {out_csv.name})")


if __name__ == "__main__":
    main()
