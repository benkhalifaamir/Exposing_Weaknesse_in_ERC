# Exposing Weaknesses in ERC

Ce document décrit comment le dépôt est organisé (dossier par dossier, fichier par
fichier) et comment relancer chaque expérience avec Ollama.

## Sommaire

- [Structure générale](#structure-générale)
- [`annotations/` — organisation par tâche](#annotations--organisation-par-tâche)
- [`codes/` — scripts](#codes--scripts)
- [Prérequis et installation](#prérequis-et-installation)
- [Reproduire les expériences avec Ollama](#ollama)

## Structure générale

```
erc/
├── annotations/                # toutes les données (brutes, augmentées, annotees, résultats)
└── codes/                      # tout le code
```

## `annotations/` — organisation par tâche

Le dossier `annotations/` est réparti en 5 sous-dossiers, chacun correspondant à une
étape du pipeline. Chaque sous-dossier (sauf `datasets_bruts/`) contient lui-même un
sous-dossier par dataset : `MELD/`, `EmoryNLP/`, `DailyDialog/`, `IEMOCAP/`.

### `datasets_bruts/`
Les 4 jeux de données de la REC dans leur format d'origine (splits train/dev/test,
variantes avec sentiment, fichiers JSON, images de distribution, etc.) — les données
sources utilisées par tous les scripts de `codes/`.

### `augmentation/`
Conversations synthétiques générées pour enrichir les classes minoritaires.

### `accord_humain/`
Tout ce qui concerne la réannotation humaine et l'accord inter-annotateurs :
- `lots_bruts_annotateurs/<Dataset>/` — un fichier par annotateur (colonne
  `Annotated_Emotion` unique).
- `annotations_consolidees/<Dataset>/` — fichiers fusionnés avec une colonne par
  annotateur (`Annotated_Emotion_1`, `_2`, ...) + `emotion_original`. Utilisés comme
  vérité terrain pour `human_agreement_analysis.py` et `llm_judge.py`.
- `indices_linguistiques_desaccord/<Dataset>/` — indices qualitatifs (`Intent_Clarity`,
  `Intent_Count`, `Context_Clarity`, `Lexical_Cue_Present`, `Punctuation`) collectés
  via [codes/app.py](codes/app.py).

### `multi_annotation_et_juge/`
- `predictions_modeles_fusionnees/<Dataset>/` — un fichier par modèle GML,
  fusionnant `emotion_original` + annotations humaines + `predicted_emotion` du
  modèle. Alimente le calcul de l'exactitude « multi-annotation » et l'accord
  inter-modèles.
- `juge_gml_par_emotion/<Dataset>/` — sorties binaires (`LLM_<émotion>` = 0/1) du
  cadre GML comme juge.
- `metriques_resultats.xlsx` — export consolidé de métriques.

### `other/`
`MELD_ERC_learner/`, `MELD_roleplay/`,
`MELD_speaker/` (jeux d'instructions au format JSONL pour des tâches de persona/
speaker-ID) et `MELD_test_openthinker_hf.py`.

## `codes/` — scripts

### Scripts 
| Fichier | Rôle |
|---|---|
| [ERC_dataset.py](codes/ERC_dataset.py) | Loaders PyTorch pour MELD/EmoryNLP/IEMOCAP/DailyDialog |
| [model.py](codes/model.py) | Modèle RoBERTa-Large + tête de classification |
| [supcon_loss.py](codes/supcon_loss.py) | Perte contrastive supervisée (SupCon) |
| [train.py](codes/train.py) / [utils.py](codes/utils.py) | Entraînement du classifieur RoBERTa. 
| [data_augment.py](codes/data_augment.py) | Extraction de matrice de transition émotionnelle (legacy, remplacée par la fonction équivalente dans `generation_augmentation.py`) |
| [code/annotation_co_general.py](codes/code/annotation_co_general.py), [code/matrice_confu_general.py](codes/code/matrice_confu_general.py), [code/autre_err_general.py](codes/code/autre_err_general.py), [code/run_pipeline.py](codes/code/run_pipeline.py) | Pipeline d'analyse d'erreurs : coloration, matrice de confusion, détection des modes d'échec linguistiques (négation, exclamation, interjection, interrogation, réplique courte) |
| [app.py](codes/app.py) + [templates/annotate.html](codes/templates/annotate.html) | Interface Flask utilisée pour collecter les indices linguistiques |
| [agents.py](codes/agents.py), [multiagent/](codes/multiagent/) | Prototype de « débat » à 2 agents — à titre indicatif seulement |


| Fichier | Rôle |
|---|---|
| [dataset_config.py](codes/dataset_config.py) | Configuration centrale des 4 datasets (étiquettes, classes minoritaires/majoritaires, personnages, thèmes, chemins) |
| [erc_common.py](codes/erc_common.py) | Chargement des dialogues, appel Ollama, extraction JSON, fusion des annotations |
| [generation_augmentation.py](codes/generation_augmentation.py) | Matrice de transition → flux émotionnel sous contraintes → génération de conversations synthétiques (zero-shot / few-shot / CoT) |
| [zero_shot_classification.py](codes/zero_shot_classification.py) | Classification zero-shot avec/sans contexte |
| [human_agreement_analysis.py](codes/human_agreement_analysis.py) | Cohen's kappa, régimes d'accord, arbre de décision sur les indices linguistiques |
| [multi_annotation_eval.py](codes/multi_annotation_eval.py) | Exactitude multi-annotation + accord inter-modèles |
| [llm_judge.py](codes/llm_judge.py) | Cadre GML comme juge (plausibilité par émotion) |


## Prérequis et installation

```bash
pip install -r codes/requirements.txt
```

Tous les scripts appellent un modèle **local via Ollama** (jamais une API cloud).
Installer [Ollama](https://ollama.com), puis récupérer les modèles utilisés, par
exemple :

```bash
ollama pull llama3.1:8b
ollama pull llama3.1:70b
ollama pull qwen2.5:32b
ollama pull mistral:7b
ollama pull gpt-oss:120b
ollama pull gemma3:27b
```

Vérifier qu'Ollama tourne (`ollama serve`, généralement déjà lancé en service) avant
d'exécuter un script.

## Reproduire les expériences avec Ollama

Tous les scripts se lancent simplement avec `python nom_du_fichier.py [options]`
depuis le dossier `codes/`.

**Génération de conversations synthétiques :**
```bash
python generation_augmentation.py --dataset MELD --model llama3.1:8b --strategy cot --n_conversations 200
python generation_augmentation.py --dataset DailyDialog --model qwen2.5:32b --strategy few-shot --n_conversations 50
```

**Classification zero-shot (avec/sans contexte) :**
```bash
python zero_shot_classification.py --dataset MELD --model llama3.1:70b
python zero_shot_classification.py --dataset MELD --model llama3.1:70b --context without
python zero_shot_classification.py --dataset EmoryNLP --model qwen2.5:32b --limit_dialogues 20   # test rapide
```
La sortie est directement compatible avec le pipeline historique d'analyse
d'erreurs (`codes/code/run_pipeline.py`) grâce aux colonnes alias `Dialogue_ID`/`Emotion`.

**Accord humain :**
```bash
python human_agreement_analysis.py --dataset MELD \
  --input ../annotations/accord_humain/annotations_consolidees/MELD/meldemotions_colored.xlsx \
  --cues  ../annotations/accord_humain/indices_linguistiques_desaccord/MELD/meld_cues_annotations.xlsx
```

**Évaluation multi-annotation :**
```bash
python multi_annotation_eval.py --dataset MELD \
  --input_dir ../annotations/multi_annotation_et_juge/predictions_modeles_fusionnees/MELD
```

**GML comme juge :**
```bash
python llm_judge.py --dataset MELD --model gemma3:27b \
  --input ../annotations/accord_humain/annotations_consolidees/MELD/meldemotions_colored.xlsx
```

Tous les scripts écrivent leurs résultats (CSV + PNG) dans `codes/files/<Dataset>/...`.


