

import pandas as pd
import argparse


def build_color_palette(labels):
    base_colors = [
        "#E56B6B", "#A3E961", "#7FEFDD", "#CDD1CA",
        "#86ACE2", "#F3A35D", "#F9F45E", "#D989B9",
        "#B57EDC", "#FFB347"
    ]
    palette = {}
    for i, lab in enumerate(labels):
        palette[lab.lower()] = base_colors[i % len(base_colors)]
    return palette


def find_dialog_col(df, user_choice=None):
    candidates = ["Dialogue_ID", "dialogue_id", "scene_id"]
    if user_choice and user_choice in df.columns:
        return user_choice
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Impossible de trouver une colonne de dialogue (Dialogue_ID / dialogue_id / scene_id)")


def combined_styling(df, true_col, pred_col, dialog_col, palette):
    styled_cells = pd.DataFrame("", index=df.index, columns=df.columns)

    for i, row in df.iterrows():
        true_emo = str(row[true_col]).strip().lower()
        pred_emo = str(row[pred_col]).strip().lower()
        if true_emo != pred_emo:
            if true_emo in palette:
                styled_cells.at[i, true_col] = f'background-color: {palette[true_emo]}'
            if pred_emo in palette:
                styled_cells.at[i, pred_col] = f'background-color: {palette[pred_emo]}'

    if dialog_col in df.columns:
        last_id = None
        highlight = False
        for i, row in df.iterrows():
            current_id = row[dialog_col]
            if current_id != last_id:
                highlight = not highlight
            color = "#8CACF1" if highlight else "#CDCACA"
            styled_cells.at[i, dialog_col] = f'background-color: {color}'
            last_id = current_id

    return styled_cells


def find_col(df, candidates):
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Fichier d'entrée (CSV ou Excel)")
    parser.add_argument("--output", type=str, required=True, help="Fichier Excel de sortie coloré")
    parser.add_argument("--true_col", type=str, default="Emotion", help="Nom de la colonne true")
    parser.add_argument("--pred_col", type=str, default="predicted_emotion", help="Nom de la colonne prédite")
    parser.add_argument("--dialog_col", type=str, default=None, help="Nom explicite de la colonne identifiant dialogue")
    parser.add_argument("--labels", nargs="+", required=True, help="Liste des émotions")
    args = parser.parse_args()

    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input, sep=None, engine="python")
    else:
        df = pd.read_excel(args.input)

    true_col_candidates = [args.true_col, "Emotion", "Label", "Labels", "true_emotion"]
    true_col = find_col(df, true_col_candidates)
    if true_col is None:
        raise ValueError(f"[ERREUR] Aucune colonne vraie émotion trouvée dans {df.columns.tolist()}")

    if "predicted_emotion_final" in df.columns:
        pred_col = "predicted_emotion_final"
    else:
        pred_col_candidates = [args.pred_col, "predicted_emotion", "Prediction", "Pred"]
        pred_col = find_col(df, pred_col_candidates)
    if pred_col is None:
        raise ValueError(f"[ERREUR] Aucune colonne prédite trouvée dans {df.columns.tolist()}")

    dialog_col = find_dialog_col(df, args.dialog_col)

    cols_to_check = [true_col, pred_col]
    before = df[dialog_col].nunique()
    invalid_dialogues = df[df[cols_to_check].isna().any(axis=1)][dialog_col].unique()
    df = df[~df[dialog_col].isin(invalid_dialogues)]
    after = df[dialog_col].nunique()
    removed = before - after
    print(f"[INFO] {removed} dialogues supprimés (valeurs NaN dans {cols_to_check})")

    palette = build_color_palette(args.labels)

    styled = df.style.apply(
        lambda _: combined_styling(df, true_col, pred_col, dialog_col, palette),
        axis=None
    )

    styled.to_excel(args.output, index=False, engine="openpyxl")
    print(f"[OK] Fichier exporté avec succès : {args.output}")


if __name__ == "__main__":
    main()
