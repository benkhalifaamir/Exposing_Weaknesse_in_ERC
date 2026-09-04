
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from dataset_config import get_config, SUPPORTED_DATASETS
from erc_common import load_conversations, call_ollama, extract_json_block, build_context_string, normalize_emotion


def build_prompt(dataset, target_speaker, target_text, context_turns, with_context):
    cfg = get_config(dataset)
    labels_str = ", ".join(cfg["labels"])

    system_prompt = (
        "You are an expert in emotion recognition in conversations. Your task is to identify "
        "the emotion expressed by the target speaker in the target utterance"
        + (", using the previous dialogue turns as context." if with_context else ".")
    )

    parts = [f"Choose exactly one label from the following list: {labels_str}."]
    if with_context and context_turns:
        parts.append("Context:\n" + build_context_string(context_turns))
    parts.append(f"Target speaker: {target_speaker}\nTarget utterance: {target_text}")
    parts.append(
        'Respond ONLY with strict JSON: '
        '{"Predicted_Emotion": "<label>", "reasoning": "<short justification>", "confidence": <float 0-1>}'
    )
    user_prompt = "\n\n".join(parts)
    return system_prompt, user_prompt


def classify_dataset(dataset, model, with_context, limit_dialogues, base_url):
    dialogues = load_conversations(dataset, split="test", max_dialogues=limit_dialogues)
    cfg = get_config(dataset)
    rows = []

    total_turns = sum(len(d) for d in dialogues)
    done = 0
    for dialogue in dialogues:
        for i, turn in enumerate(dialogue):
            context_turns = dialogue[:i] if with_context else []
            system_prompt, user_prompt = build_prompt(
                dataset, turn["speaker"], turn["text"], context_turns, with_context
            )
            raw = call_ollama(user_prompt, model=model, system_prompt=system_prompt,
                               temperature=0.0, base_url=base_url)
            parsed = extract_json_block(raw) or {}
            pred = normalize_emotion(dataset, parsed.get("Predicted_Emotion", "")) if isinstance(parsed, dict) else ""
            if pred not in cfg["labels"]:
                pred = cfg["majority_label"]

            rows.append({
                "dataset": dataset,
                "dialog_id": turn["dialog_id"],
                "Dialogue_ID": turn["dialog_id"],
                "turn_id": turn["turn_id"],
                "speaker": turn["speaker"],
                "utterance": turn["text"],
                "emotion_original": turn["emotion"],
                "Emotion": turn["emotion"],
                "emotion": turn["emotion"],
                "predicted_emotion": pred,
                "reasoning": parsed.get("reasoning", "") if isinstance(parsed, dict) else "",
                "confidence": parsed.get("confidence", None) if isinstance(parsed, dict) else None,
                "context": "with" if with_context else "without",
                "model": model,
            })
            done += 1
        if done % 50 < len(dialogue):
            print(f"  ... {done}/{total_turns} repliques classees")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Classification zero-shot des emotions (chapitre 3)")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model", default="llama3.1:70b", help="Modele Ollama (ex: llama3.1:70b, qwen2.5:32b, gpt-oss:120b, mistral:7b)")
    parser.add_argument("--context", default="with", choices=["with", "without"])
    parser.add_argument("--limit_dialogues", type=int, default=None, help="Limite le nb de dialogues (tests rapides)")
    parser.add_argument("--base_url", default="http://localhost:11434")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"[1/2] Classification zero-shot {args.dataset} avec {args.model} (contexte={args.context})...")
    df = classify_dataset(args.dataset, args.model, args.context == "with", args.limit_dialogues, args.base_url)

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "files" / args.dataset / args.model.replace(":", "-") /
        f"{args.context}_context" / "predictions.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[2/2] {len(df)} predictions -> {out_path}")
    report = classification_report(df["emotion_original"], df["predicted_emotion"], zero_division=0, output_dict=True)
    print(f"  Macro-F1    : {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted-F1 : {report['weighted avg']['f1-score']:.3f}")
    for label in get_config(args.dataset)["labels"]:
        if label in report:
            print(f"    {label:12s} F1={report[label]['f1-score']:.3f}")


if __name__ == "__main__":
    main()
