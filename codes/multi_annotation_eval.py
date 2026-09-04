
import argparse
import glob
from pathlib import Path

import pandas as pd

from dataset_config import get_config, SUPPORTED_DATASETS
from erc_common import load_reannotation, plausible_set, normalize_emotion


def find_prediction_column(df):
    for cand in ["predicted_emotion", "Predicted_Emotion", "prediction"]:
        if cand in df.columns:
            return cand
    raise ValueError(f"Aucune colonne de prediction trouvee parmi {list(df.columns)}")


def soft_agreement_accuracy(path, dataset):
    df, ann_cols = load_reannotation(path, dataset)
    pred_col = find_prediction_column(df)
    df[pred_col] = df[pred_col].apply(lambda v: normalize_emotion(dataset, v) if pd.notna(v) else v)

    correct = 0
    for _, row in df.iterrows():
        plausible = plausible_set(row, ann_cols)
        if row[pred_col] in plausible:
            correct += 1
    accuracy = correct / len(df) if len(df) else float("nan")
    return accuracy, df, ann_cols, pred_col


def inter_model_agreement(dataset, model_files):
    cfg = get_config(dataset)
    neutral_positive = set(cfg["positive_labels"]) | {cfg["majority_label"]}

    per_model_preds = {}
    ann_cols_ref, key_cols_ref, base_df = None, None, None

    for path in model_files:
        df, ann_cols = load_reannotation(path, dataset)
        pred_col = find_prediction_column(df)
        df[pred_col] = df[pred_col].apply(lambda v: normalize_emotion(dataset, v) if pd.notna(v) else v)
        model_name = Path(path).stem
        per_model_preds[model_name] = df[pred_col].reset_index(drop=True)
        if base_df is None:
            base_df = df.reset_index(drop=True)
            ann_cols_ref = ann_cols

    n_items = len(base_df)
    n_agree, n_agree_neutral_pos = 0, 0
    for i in range(n_items):
        preds_i = [per_model_preds[m].iloc[i] for m in per_model_preds if i < len(per_model_preds[m])]
        if not preds_i:
            continue
        counts = pd.Series(preds_i).value_counts()
        majority_label = counts.index[0]
        majority_frac = counts.iloc[0] / len(preds_i)
        if majority_frac > 0.5:
            n_agree += 1
            if majority_label in neutral_positive:
                n_agree_neutral_pos += 1

    agreement_rate = n_agree / n_items if n_items else float("nan")
    neutral_positive_rate = n_agree_neutral_pos / n_agree if n_agree else float("nan")
    return agreement_rate, neutral_positive_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluation multi-annotation (chapitre 5.1)")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--input", default=None, help="Un seul fichier de predictions fusionnees")
    parser.add_argument("--input_dir", default=None, help="Dossier contenant un fichier par modele")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Fournir --input (un modele) ou --input_dir (plusieurs modeles)")

    results = []
    if args.input:
        acc, df, ann_cols, pred_col = soft_agreement_accuracy(args.input, args.dataset)
        model_name = Path(args.input).stem
        print(f"{args.dataset} {model_name} :")
        print(f"  Soft agreement accuracy : {acc:.2f}")
        results.append({"dataset": args.dataset, "model": model_name, "soft_agreement_accuracy": acc})

    if args.input_dir:
        model_files = sorted(glob.glob(str(Path(args.input_dir) / "*.csv")))
        if not model_files:
            model_files = sorted(glob.glob(str(Path(args.input_dir) / "*.xlsx")))
        print(f"{len(model_files)} fichiers modele trouves dans {args.input_dir}")

        for path in model_files:
            acc, *_ = soft_agreement_accuracy(path, args.dataset)
            model_name = Path(path).stem
            print(f"{args.dataset} {model_name} :")
            print(f"  Soft agreement accuracy : {acc:.2f}")
            results.append({"dataset": args.dataset, "model": model_name, "soft_agreement_accuracy": acc})

        if len(model_files) > 1:
            agreement_rate, neutral_positive_rate = inter_model_agreement(args.dataset, model_files)
            print(f"\n{args.dataset} -- accord inter-modeles :")
            print(f"  Accord inter-modeles (%)        : {100 * agreement_rate:.0f}")
            print(f"  Accord sur neutre/positif (%)   : {100 * neutral_positive_rate:.0f}")
            results.append({
                "dataset": args.dataset, "model": "ALL_MODELS_INTER_AGREEMENT",
                "inter_model_agreement_pct": 100 * agreement_rate,
                "agreement_on_neutral_or_positive_pct": 100 * neutral_positive_rate,
            })

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "files" / args.dataset / "chapitre5_multi_annotation" / "results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResultats sauvegardes dans {out_path}")


if __name__ == "__main__":
    main()
