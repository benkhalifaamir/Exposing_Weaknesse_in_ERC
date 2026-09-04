
from pathlib import Path

CODES_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = CODES_DIR.parent / "annotations"
RAW_DIR = ANNOTATIONS_DIR / "datasets_bruts"

SUPPORTED_DATASETS = ["MELD", "EmoryNLP", "DailyDialog", "IEMOCAP"]

FRIENDS_CHARACTERS = {
    "Ross": "Ross is intellectual, awkward, and prone to over-explaining and jealousy.",
    "Rachel": "Rachel is fashion-conscious, warm, and quick to sarcasm when annoyed.",
    "Monica": "Monica is competitive, controlling about order and cleanliness, and fiercely loyal.",
    "Chandler": "Chandler deflects emotion with sarcasm and self-deprecating jokes.",
    "Joey": "Joey is simple-hearted, food- and romance-obsessed, but loyal to his friends.",
    "Phoebe": "Phoebe is quirky, blunt, and unpredictably philosophical.",
}

MELD_TOPICS = ["romantic relationships", "work and career problems", "roommate conflicts",
               "family gatherings", "friendship drama", "dating mishaps"]
EMORYNLP_TOPICS = ["breakups and reconciliations", "job interviews", "apartment troubles",
                    "wedding planning", "sibling rivalry", "unexpected reunions"]
DAILYDIALOG_TOPICS = ["ordering food at a restaurant", "asking for directions", "a job interview",
                       "planning a trip", "a doctor's appointment", "returning a faulty product",
                       "catching up with an old friend", "negotiating a price at a market"]
IEMOCAP_TOPICS = ["a difficult breakup", "an argument about money", "comforting a friend in distress",
                   "a stressful work deadline", "a surprise announcement", "a disagreement between roommates"]

DATASETS = {
    "MELD": {
        "labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        "minority_labels": ["fear", "disgust"],
        "majority_label": "neutral",
        "positive_labels": ["joy"],
        "label_map": {},
        "has_fixed_characters": True,
        "characters": FRIENDS_CHARACTERS,
        "topics": MELD_TOPICS,
        "raw_test_path": CODES_DIR / "test_data.csv",
        "raw_train_path": RAW_DIR / "MELD" / "MELD_ERCdata" / "train.json",
        "raw_sep": ",",
        "raw_cols": {"dialog_id": "Dialogue_ID", "speaker": "Speaker", "text": "Utterance", "emotion": "Emotion"},
    },
    "EmoryNLP": {
        "labels": ["joyful", "mad", "neutral", "peaceful", "powerful", "sad", "scared"],
        "minority_labels": ["peaceful", "powerful", "scared"],
        "majority_label": "neutral",
        "positive_labels": ["joyful", "peaceful", "powerful"],
        "label_map": {},
        "has_fixed_characters": True,
        "characters": FRIENDS_CHARACTERS,
        "topics": EMORYNLP_TOPICS,
        "raw_test_path": RAW_DIR / "EmoryNLP" / "test.csv",
        "raw_train_path": RAW_DIR / "EmoryNLP" / "train.csv",
        "raw_sep": ";",
        "raw_cols": {"dialog_id": "scene_id", "speaker": "speakers", "text": "transcript", "emotion": "emotion"},
    },
    "DailyDialog": {
        "labels": ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
        "minority_labels": ["fear", "disgust"],
        "majority_label": "neutral",
        "positive_labels": ["happiness"],
        "label_map": {"no emotion": "neutral"},
        "has_fixed_characters": False,
        "characters": None,
        "topics": DAILYDIALOG_TOPICS,
        "raw_test_path": RAW_DIR / "DailyDialog" / "data" / "dailydialog_with_sentiment.csv",
        "raw_sep": ",",
        "raw_cols": {"dialog_id": "dialogue_id", "speaker": "speaker", "text": "utterance", "emotion": "emotion"},
        "split_col": "data_split",
    },
    "IEMOCAP": {
        "labels": ["angry", "excited", "frustrated", "happy", "neutral", "sad"],
        "minority_labels": ["frustrated", "excited"],
        "majority_label": "neutral",
        "positive_labels": ["excited", "happy"],
        "label_map": {"ang": "angry", "exc": "excited", "fru": "frustrated",
                       "hap": "happy", "neu": "neutral", "sad": "sad"},
        "has_fixed_characters": False,
        "characters": None,
        "topics": IEMOCAP_TOPICS,
        "raw_test_path": RAW_DIR / "IEMOCAP" / "test.csv",
        "raw_train_path": RAW_DIR / "IEMOCAP" / "train.csv",
        "raw_sep": ",",
        "raw_cols": {"dialog_id": "scene", "speaker": "speaker", "text": "text", "emotion": "label"},
    },
}


def get_config(dataset):
    if dataset == "MELD":
        return DATASETS["MELD"]
    elif dataset == "EmoryNLP":
        return DATASETS["EmoryNLP"]
    elif dataset == "DailyDialog":
        return DATASETS["DailyDialog"]
    elif dataset == "IEMOCAP":
        return DATASETS["IEMOCAP"]
    else:
        raise ValueError(f"Dataset inconnu : {dataset}. Choix possibles : {SUPPORTED_DATASETS}")
