
import json
import random
import re

import numpy as np
import pandas as pd

from dataset_config import get_config
from ollama_engine import OllamaEngine


def _rename_and_clean(df, cfg):
    df = df.rename(columns={
        cfg["raw_cols"]["dialog_id"]: "dialog_id",
        cfg["raw_cols"]["speaker"]: "speaker",
        cfg["raw_cols"]["text"]: "text",
        cfg["raw_cols"]["emotion"]: "emotion",
    })
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower().replace(cfg["label_map"])
    return df


def load_conversations(dataset, split="test", max_dialogues=None, seed=42):
    cfg = get_config(dataset)

    if dataset == "MELD":
        if split == "test":
            df = pd.read_csv(cfg["raw_test_path"])
            df = _rename_and_clean(df, cfg)
            df = df.sort_values(["dialog_id", "Utterance_ID"])
        else:
            rows = []
            with open(cfg["raw_train_path"], encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rows.append({
                        "dialog_id": rec["dialogue_id"],
                        "turn_order": rec["utterance_id"],
                        "speaker": rec["speaker"],
                        "text": rec["utterance"],
                        "emotion": str(rec["emotion"]).strip().lower(),
                    })
            df = pd.DataFrame(rows).sort_values(["dialog_id", "turn_order"])

    elif dataset == "EmoryNLP":
        path = cfg["raw_test_path"] if split == "test" else cfg["raw_train_path"]
        df = pd.read_csv(path, sep=cfg["raw_sep"])
        df = _rename_and_clean(df, cfg)
        df = df.sort_values(["dialog_id", "utterance_id"])

    elif dataset == "DailyDialog":
        df = pd.read_csv(cfg["raw_test_path"])
        df = df[df[cfg["split_col"]] == split].reset_index(drop=True)
        df = _rename_and_clean(df, cfg)

    elif dataset == "IEMOCAP":
        path = cfg["raw_test_path"] if split == "test" else cfg["raw_train_path"]
        df = pd.read_csv(path)
        df = _rename_and_clean(df, cfg)
        df = df[df["emotion"].isin(cfg["labels"])].reset_index(drop=True)

    else:
        raise ValueError(f"Dataset inconnu : {dataset}")

    dialogues = []
    for dialog_id, group in df.groupby("dialog_id", sort=False):
        turns = []
        for turn_id, row in enumerate(group.itertuples(index=False)):
            turns.append({
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "speaker": str(getattr(row, "speaker")),
                "text": str(getattr(row, "text")),
                "emotion": str(getattr(row, "emotion")),
            })
        dialogues.append(turns)

    rng = random.Random(seed)
    if max_dialogues is not None and len(dialogues) > max_dialogues:
        dialogues = rng.sample(dialogues, max_dialogues)

    return dialogues


def normalize_emotion(dataset, raw_value):
    cfg = get_config(dataset)
    val = str(raw_value).strip().lower()
    val = cfg["label_map"].get(val, val)
    aliases = {
        "MELD": {"happy": "joy"},
        "EmoryNLP": {"joy": "joyful", "angry": "mad", "scary": "scared"},
        "DailyDialog": {"happy": "happiness", "joy": "happiness"},
        "IEMOCAP": {"ang": "angry", "hap": "happy", "neu": "neutral", "sad ": "sad"},
    }
    val = aliases.get(dataset, {}).get(val, val)
    return val if val in cfg["labels"] else val


def pick_characters(dataset, n_speakers, seed=None):
    cfg = get_config(dataset)
    rng = random.Random(seed)
    if dataset in ("MELD", "EmoryNLP") and cfg["has_fixed_characters"]:
        names = rng.sample(list(cfg["characters"].keys()), min(n_speakers, len(cfg["characters"])))
        return {name: cfg["characters"][name] for name in names}
    else:
        generic = ["Speaker_A", "Speaker_B", "Speaker_C"][:n_speakers]
        return {name: "A generic conversational participant with a distinct communication style."
                for name in generic}


def pick_topic(dataset, seed=None):
    cfg = get_config(dataset)
    rng = random.Random(seed)
    return rng.choice(cfg["topics"])


_engine_cache = {}


def get_engine(base_url="http://localhost:11434"):
    if base_url not in _engine_cache:
        _engine_cache[base_url] = OllamaEngine(base_url=base_url)
    return _engine_cache[base_url]


def call_ollama(prompt, model, system_prompt=None, temperature=0.3, base_url="http://localhost:11434"):
    engine = get_engine(base_url)
    return engine.generate(prompt, model=model, temperature=temperature, system_prompt=system_prompt)


def extract_json_block(text):
    if text is None:
        return None

    m = re.search(r"###BEGIN_JSON###(.*?)###END_JSON###", text, re.DOTALL)
    candidate = m.group(1).strip() if m else text.strip()

    candidate = re.sub(r"^```(json)?|```$", "", candidate.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for opener, closer in [("[", "]"), ("{", "}")]:
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def build_context_string(turns):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def find_annotator_columns(df):
    cols = []
    if "emotion_original" in df.columns:
        cols.append("emotion_original")
    elif "Emotion_original" in df.columns:
        cols.append("Emotion_original")
    ann_cols = sorted([c for c in df.columns if re.match(r"^Annotated_Emotion", c, re.IGNORECASE)])
    return cols + ann_cols


def load_reannotation(path, dataset):
    if str(path).endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    ann_cols = find_annotator_columns(df)
    if len(ann_cols) < 2:
        raise ValueError(f"Moins de 2 colonnes d'annotation detectees dans {path} : {list(df.columns)}")

    for c in ann_cols:
        df[c] = df[c].apply(lambda v: normalize_emotion(dataset, v) if pd.notna(v) else np.nan)

    return df, ann_cols


def plausible_set(row, ann_cols):
    return {row[c] for c in ann_cols if pd.notna(row[c])}
