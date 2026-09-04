
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_config import get_config, SUPPORTED_DATASETS
from erc_common import (
    load_reannotation, plausible_set, load_conversations,
    call_ollama, build_context_string,
)


def build_context_lookup(dataset):
    lookup = {}
    for dialogue in load_conversations(dataset, split="test"):
        for i, turn in enumerate(dialogue):
            key = (str(turn["dialog_id"]), turn["text"].strip().lower())
            lookup[key] = dialogue[:i]
    return lookup


def judge_prompt(dataset, target_speaker, target_text, context_turns, candidate_emotion):
    system_prompt = (
        "You are an expert in emotion recognition in conversations. Your task is to determine "
        "whether a given emotion label is plausible for a target utterance, given the dialogue context."
    )
    parts = []
    if context_turns:
        parts.append("Dialogue context:\n" + build_context_string(context_turns))
    parts.append(f"Target speaker: {target_speaker}\nTarget utterance: {target_text}")
    parts.append(f"Candidate emotion: {candidate_emotion}")
    parts.append(
        "Given the context and the target utterance, answer whether the candidate emotion "
        "is plausible. Respond with exactly one word: Yes or No."
    )
    return system_prompt, "\n\n".join(parts)


def parse_yes_no(raw):
    if raw is None:
        return 0
    return 1 if re.search(r"\byes\b", raw.strip().lower()) else 0


def run_judge(dataset, model, input_path, base_url, limit=None):
    cfg = get_config(dataset)
    labels = cfg["labels"]

    df, ann_cols = load_reannotation(input_path, dataset)
    if limit:
        df = df.head(limit)

    context_lookup = build_context_lookup(dataset)

    judge_cols = {lab: [] for lab in labels}
    n_context_found = 0
    for _, row in df.iterrows():
        dialog_id = str(row.get("dialog_id", row.get("scene_id", "")))
        text_key = str(row.get("utterance", "")).strip().lower()
        context_turns = context_lookup.get((dialog_id, text_key), [])
        n_context_found += int(bool(context_turns))
        speaker = row.get("speaker", "?")

        for lab in labels:
            system_prompt, user_prompt = judge_prompt(dataset, speaker, row.get("utterance", ""), context_turns, lab)
            raw = call_ollama(user_prompt, model=model, system_prompt=system_prompt,
                               temperature=0.3, base_url=base_url)
            judge_cols[lab].append(parse_yes_no(raw))

    for lab in labels:
        df[f"LLM_{lab}"] = judge_cols[lab]

    print(f"  Contexte retrouve pour {n_context_found}/{len(df)} enonces "
          f"(les autres sont juges sans contexte precedent).")
    return df, ann_cols, labels


def compute_metrics(df, ann_cols, labels):
    y_judge, y_human = [], []
    tp = fp = tn = fn = 0
    for _, row in df.iterrows():
        plausible = plausible_set(row, ann_cols)
        for lab in labels:
            judge_says_yes = bool(row[f"LLM_{lab}"])
            human_says_yes = lab in plausible
            y_judge.append(int(judge_says_yes))
            y_human.append(int(human_says_yes))
            if judge_says_yes and human_says_yes:
                tp += 1
            elif judge_says_yes and not human_says_yes:
                fp += 1
            elif not judge_says_yes and human_says_yes:
                fn += 1
            else:
                tn += 1

    n_total = tp + fp + tn + fn
    compatibility = (tp + tn) / n_total if n_total else float("nan")
    hard_cases = tp + fp + fn
    exactitude = tp / hard_cases if hard_cases else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    tpn = fp / (fp + tn) if (fp + tn) else float("nan")
    kappa = cohen_kappa_score(y_judge, y_human) if len(set(y_judge)) > 1 and len(set(y_human)) > 1 else float("nan")

    return {
        "cohen_kappa": kappa, "compatibilite": compatibility, "exactitude": exactitude,
        "rappel": recall, "TPN": tpn, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def plot_confusion(metrics, dataset, model, out_dir):
    total = metrics["TP"] + metrics["FP"] + metrics["TN"] + metrics["FN"]
    if total == 0:
        return
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]) / total * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=["Non plausible", "Plausible"], yticklabels=["Non plausible", "Plausible"], ax=ax)
    ax.set_xlabel("Decision du juge")
    ax.set_ylabel("Verite terrain humaine")
    ax.set_title(f"Juge GML {model} sur {dataset} (%)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{dataset}_{model.replace(':', '-')}_judge_confusion.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Cadre GML comme juge (chapitre 5.2)")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model", default="gemma3:27b", help="Modele Ollama juge (ex: gemma3:27b, llama3.1:70b, qwen2.5:32b)")
    parser.add_argument("--input", required=True, help="Fichier de reannotation fusionne")
    parser.add_argument("--limit", type=int, default=None, help="Limite le nb d'enonces (tests rapides)")
    parser.add_argument("--base_url", default="http://localhost:11434")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "files" / args.dataset / "chapitre5_juge_gml"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Jugement de plausibilite par emotion ({args.dataset}, juge={args.model})...")
    df, ann_cols, labels = run_judge(args.dataset, args.model, args.input, args.base_url, args.limit)

    print("[2/3] Calcul des metriques (Tableau 5.5)...")
    metrics = compute_metrics(df, ann_cols, labels)
    print(f"  Cohen's kappa : {metrics['cohen_kappa']:.2f}")
    print(f"  Compatibilite : {metrics['compatibilite']:.2f}")
    print(f"  Exactitude    : {metrics['exactitude']:.2f}")
    print(f"  Rappel        : {metrics['rappel']:.2f}")
    print(f"  TPN           : {metrics['TPN']:.2f}")

    plot_confusion(metrics, args.dataset, args.model, out_dir)

    out_csv = out_dir / f"{args.dataset}_{args.model.replace(':', '-')}_judge_per_emotion.csv"
    df.to_csv(out_csv, index=False)
    pd.DataFrame([{"dataset": args.dataset, "model": args.model, **metrics}]).to_csv(
        out_dir / f"{args.dataset}_{args.model.replace(':', '-')}_judge_metrics.csv", index=False
    )
    print(f"[3/3] Resultats sauvegardes dans {out_dir}")


if __name__ == "__main__":
    main()
