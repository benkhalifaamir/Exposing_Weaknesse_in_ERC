

import os
import subprocess
import argparse
from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_DATASETS = [
    "DDwithcontextgpt.csv",
    "DDwithcontextllama70b 1.csv",
    "DDwithcontext_mistral7b 1.csv",
    "DDwithcontext_qwen32b 1.csv",
    "ddwithoutcontextgpt.csv",
    "ddwithoutcontextllama70 1.csv",
    "ddwithoutcontext_mistral7b 1.csv",
    "ddwithoutcontext_qwen32b 1.csv",
    "emorynlpwithcontextgptoss.csv",
    "emorynlpwithcontextllama70b 1.csv",
    "emorynlpwithcontext_mistral7b 1.csv",
    "emorynlpwithcontext_qwen32b 1.csv",
    "emorynlpwithoutcontextgptoss.csv",
    "emorynlpwithoutcontextllama70b 1.csv",
    "emorynlpwithoutcontext_mistral7b 1.csv",
    "emorynlpwithoutcontext_qwen32b 1.csv",
    "Meldwithcontextgptoss.csv",
    "MeldwithcontextLLAMA70b 1.csv",
    "Meldwithcontextmistral 2.csv",
    "Meldwithcontext_qwen32b 1.csv",
    "meldwithoutcontextLLAMA70b 1.csv",
    "meldwithoutcontextmistral 2.csv",
    "meldwithoutcontext_gptoss.csv",
    "meldwithoutcontext_qwen32b 1.csv"
]

LABELS = {
    "MELD": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "EmoryNLP": ["joyful", "mad", "neutral", "peaceful", "powerful", "sad", "scared"],
    "DailyDialog": ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
}

COLUMNS = {
    "MELD": {"true_col": "Emotion", "pred_col": "predicted_emotion"},
    "EmoryNLP": {"true_col": "emotion", "pred_col": "predicted_emotion"},
    "DailyDialog": {"true_col": "emotion", "pred_col": "predicted_emotion"},
}


def parse_dataset_name(filename):
    name = Path(filename).stem.lower()

    dataset_aliases = {
        "meld": "MELD",
        "emory": "EmoryNLP",
        "emorynlp": "EmoryNLP",
        "dd": "DailyDialog",
        "dailydialog": "DailyDialog"
    }

    dataset = None
    for alias, canon in dataset_aliases.items():
        if alias in name:
            dataset = canon
            break

    if dataset is None:
        raise ValueError(f"Dataset non reconnu dans {filename}")

    context = "with_context" if "withcontext" in name else "without_context"

    if "mistral" in name:
        model = "Mistral"
    elif "llama" in name:
        model = "LLaMA70B"
    elif "qwen" in name:
        model = "Qwen32B"
    elif "gpt" in name:
        model = "GPT-OSS"
    else:
        model = "Unknown"

    return dataset, context, model

def run_command(script_name, file_name, cmd, cwd=None):
    try:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[RUN] {script_name} sur {file_name} ... OK")
    except subprocess.CalledProcessError:
        print(f"[RUN] {script_name} sur {file_name} ... ERREUR")


def normalize_labels(labels):
    return [lbl.strip().lower() for lbl in labels]


def process_dataset(file_path, out_root="results"):
    dataset, context, model = parse_dataset_name(file_path)
    out_dir = Path(out_root) / dataset / model / context
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = COLUMNS[dataset]
    labels = normalize_labels(LABELS[dataset])

    colored_path = out_dir / f"{Path(file_path).stem}_colored.xlsx"

    run_command("annotation_co_general.py", Path(file_path).name, [
        "python", str(SCRIPT_DIR / "annotation_co_general.py"),
        "--input", file_path,
        "--true_col", cols["true_col"],
        "--pred_col", cols["pred_col"],
        "--labels", *labels,
        "--output", str(colored_path)
    ])

    run_command("matrice_confu_general.py", Path(file_path).name, [
        "python", str(SCRIPT_DIR / "matrice_confu_general.py"),
        "--input", str(colored_path),
        "--true_col", cols["true_col"],
        "--pred_col", cols["pred_col"],
        "--labels", *labels,
        "--output_dir", str(out_dir)
    ])

    run_command("autre_err_general.py", Path(file_path).name, [
        "python", str(SCRIPT_DIR / "autre_err_general.py"),
        "--input", str(colored_path),
        "--true_col", cols["true_col"],
        "--pred_col", cols["pred_col"],
        "--output_dir", str(out_dir)
    ])


    print(f"[OK] Résultats sauvegardés dans {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Liste des fichiers datasets à traiter")
    parser.add_argument("--all", action="store_true",
                        help="Traiter tous les fichiers du dossier datasets/")
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Répertoire où chercher les datasets si --all est activé")
    parser.add_argument("--out_root", type=str, default="results",
                        help="Répertoire racine pour les sorties")
    args = parser.parse_args()

    if args.all:
        dataset_dir = Path(args.dataset_dir)
        files = list(dataset_dir.glob("*.csv")) + list(dataset_dir.glob("*.xlsx"))
    else:
        files = args.datasets if args.datasets else DEFAULT_DATASETS
        files = [Path(f) for f in files]

    print(f"[INFO] {len(files)} fichiers à traiter")

    for file in files:
        process_dataset(str(file), out_root=args.out_root)
