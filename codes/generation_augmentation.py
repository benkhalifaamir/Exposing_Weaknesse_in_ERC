
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_config import get_config, SUPPORTED_DATASETS
from erc_common import (
    load_conversations, call_ollama, extract_json_block,
    pick_characters, pick_topic,
)

FEW_SHOT_EXAMPLE = (
    'Emotion flow: [Sadness -> Anger -> Neutral]. Topic: work.\n'
    '[{"speaker": "Ross", "text": "I just found out I did not get the promotion.", "emotion": "sadness"},\n'
    ' {"speaker": "Rachel", "text": "That is completely unfair, you deserved it!", "emotion": "anger"},\n'
    ' {"speaker": "Ross", "text": "I know, but I will figure it out.", "emotion": "neutral"}]'
)


def build_transition_matrix(dataset):
    cfg = get_config(dataset)
    labels = cfg["labels"]
    idx = {lab: i for i, lab in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)))

    dialogues = load_conversations(dataset, split="train")
    for dialogue in dialogues:
        for t in range(len(dialogue) - 1):
            e_cur = dialogue[t]["emotion"]
            e_next = dialogue[t + 1]["emotion"]
            if e_cur in idx and e_next in idx:
                counts[idx[e_cur]][idx[e_next]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = counts / row_sums
    return transition_matrix, labels


def sample_emotion_flow(dataset, transition_matrix, labels, n_turns=8, max_attempts=200, rng=None):
    cfg = get_config(dataset)
    minority = set(cfg["minority_labels"])
    majority = cfg["majority_label"]
    idx = {lab: i for i, lab in enumerate(labels)}
    rng = rng or np.random.default_rng()

    for _ in range(max_attempts):
        start = rng.choice(labels)
        flow = [start]
        for _ in range(n_turns - 1):
            probs = transition_matrix[idx[flow[-1]]]
            if probs.sum() == 0:
                nxt = rng.choice(labels)
            else:
                nxt = rng.choice(labels, p=probs / probs.sum())
            flow.append(nxt)

        n_minority = sum(1 for e in flow if e in minority)
        n_majority = sum(1 for e in flow if e == majority)
        if n_minority >= 2 and n_majority <= 1:
            return flow

    flow = list(rng.choice(list(minority), size=min(2, len(minority)), replace=True))
    while len(flow) < n_turns:
        flow.append(rng.choice([l for l in labels if l != majority]))
    rng.shuffle(flow)
    return flow[:n_turns]


def build_prompt(dataset, emotion_flow, characters, topic, strategy):
    char_desc = "; ".join(f"{name}: {desc}" for name, desc in characters.items())
    flow_str = " -> ".join(emotion_flow)
    output_format = (
        '###BEGIN_JSON###\n'
        '[{"speaker": "...", "text": "...", "emotion": "..."}]\n'
        '###END_JSON###'
    )

    if strategy == "zero-shot":
        system_prompt = (
            "You are an expert dialogue generator. Your task is to generate a realistic "
            "and emotionally coherent multi-turn conversation between characters, strictly "
            "following the given emotional trajectory."
        )
        user_prompt = (
            f"The conversation must follow this exact emotion flow: {flow_str}. "
            f"Each turn must reflect the assigned emotion. Do not include any emotion not listed in the flow.\n"
            f"Topic: {topic}. Characters: {char_desc}.\n"
            f"Output format:\n{output_format}"
        )

    elif strategy == "few-shot":
        system_prompt = (
            "You are an expert dialogue generator. Your task is to generate a realistic and "
            "emotionally coherent multi-turn conversation strictly following the given emotional "
            "trajectory. Below is an example to guide your generation."
        )
        user_prompt = (
            f"Example:\n{FEW_SHOT_EXAMPLE}\n\n"
            f"Now generate a conversation for: emotion flow {flow_str}, topic {topic}, characters {char_desc}.\n"
            f"Output format:\n{output_format}"
        )

    elif strategy == "cot":
        system_prompt = (
            "You are simulating emotionally rich dialogues between different personas. Your task is "
            "to generate a realistic conversation where each character's emotional behaviour and "
            "personality are faithfully preserved."
        )
        user_prompt = (
            f"Step 1: Select characters relevant to the topic {topic}. Use only characters from the "
            f"provided list: {char_desc}.\n"
            f"Step 2: Follow the predefined emotion flow strictly: {flow_str}. Do not include emotions "
            f"not listed. Each turn must match the assigned emotion.\n"
            f"Step 3: Write a natural multi-turn dialogue consistent with the emotion progression and "
            f"character traits. Maintain language consistent with the personality, emotionally plausible "
            f"transitions, and avoid verbosity or overly formal phrasing.\n"
            f"Output format:\n{output_format}"
        )
    else:
        raise ValueError("strategy doit etre 'zero-shot', 'few-shot' ou 'cot'")

    return system_prompt, user_prompt


def generate_one_conversation(dataset, transition_matrix, labels, model, strategy, n_turns, rng):
    cfg = get_config(dataset)
    n_speakers = 2 if not cfg["has_fixed_characters"] else rng.integers(2, 4)
    characters = pick_characters(dataset, int(n_speakers), seed=int(rng.integers(0, 10**6)))
    topic = pick_topic(dataset, seed=int(rng.integers(0, 10**6)))
    flow = sample_emotion_flow(dataset, transition_matrix, labels, n_turns=n_turns, rng=rng)

    system_prompt, user_prompt = build_prompt(dataset, flow, characters, topic, strategy)
    raw = call_ollama(user_prompt, model=model, system_prompt=system_prompt, temperature=0.3)
    parsed = extract_json_block(raw)

    turns = []
    if isinstance(parsed, list):
        for i, turn in enumerate(parsed):
            turns.append({
                "speaker": turn.get("speaker", "?"),
                "text": turn.get("text", ""),
                "emotion_generated": str(turn.get("emotion", "")).strip().lower(),
                "emotion_target": flow[i] if i < len(flow) else None,
            })
    return {
        "topic": topic,
        "characters": list(characters.keys()),
        "emotion_flow_target": flow,
        "turns": turns,
        "raw_response": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Generation de conversations synthetiques (chapitre 2)")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model", default="llama3.1:8b", help="Modele Ollama (ex: llama3.1:8b, qwen2.5:32b)")
    parser.add_argument("--strategy", default="cot", choices=["zero-shot", "few-shot", "cot"])
    parser.add_argument("--n_conversations", type=int, default=200)
    parser.add_argument("--n_turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="CSV de sortie (defaut: codes/generated/<Dataset>/...)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[1/3] Construction de la matrice de transition ({args.dataset}, train)...")
    transition_matrix, labels = build_transition_matrix(args.dataset)

    print(f"[2/3] Generation de {args.n_conversations} conversations via Ollama "
          f"({args.model}, strategie={args.strategy})...")
    rows = []
    n_ok, n_flow_respected_first2, n_flow_respected_rest = 0, 0, 0
    for conv_id in range(args.n_conversations):
        result = generate_one_conversation(args.dataset, transition_matrix, labels,
                                            args.model, args.strategy, args.n_turns, rng)
        if not result["turns"]:
            print(f"  [WARN] conversation {conv_id} : JSON non parsable, ignoree")
            continue
        n_ok += 1
        for i, turn in enumerate(result["turns"]):
            respected = turn["emotion_generated"] == turn["emotion_target"]
            if i < 2:
                n_flow_respected_first2 += int(respected)
            else:
                n_flow_respected_rest += int(respected)
            rows.append({
                "conversation_id": conv_id,
                "turn": i,
                "speaker": turn["speaker"],
                "text": turn["text"],
                "emotion": turn["emotion_generated"],
                "emotion_target": turn["emotion_target"],
                "flow_respected": respected,
                "topic": result["topic"],
                "strategy": args.strategy,
                "model": args.model,
            })
        if (conv_id + 1) % 10 == 0:
            print(f"  ... {conv_id + 1}/{args.n_conversations} generees ({n_ok} valides)")

    df = pd.DataFrame(rows)
    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "generated" / args.dataset / f"synthetic_{args.strategy}_{args.model.replace(':', '-')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[3/3] Termine : {n_ok}/{args.n_conversations} conversations valides -> {out_path}")
    total_first2 = n_ok * min(2, args.n_turns)
    total_rest = len(df) - total_first2
    if total_first2:
        print(f"  Respect du flux, tours 1-2  : {100 * n_flow_respected_first2 / total_first2:.1f}%")
    if total_rest > 0:
        print(f"  Respect du flux, tours 3+   : {100 * n_flow_respected_rest / total_rest:.1f}%")


if __name__ == "__main__":
    main()
