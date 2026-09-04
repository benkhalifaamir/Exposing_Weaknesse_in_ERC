

import os
import io
import base64
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import seaborn as sns


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_features(df, utt_col="Utterance"):
    text = df[utt_col].fillna("").astype(str)

    df["contient_point_interrogation"] = text.str.contains(r"\?", regex=True)
    df["contient_point_exclamation"]   = text.str.contains(r"\!", regex=True)

    df["nb_mots"] = text.str.split().str.len()
    df["tranche_longueur"] = pd.cut(
        df["nb_mots"], bins=[-1, 3, 15, 10**9],
        labels=["≤3 mots", "4–15 mots", ">15 mots"]
    )

    neg_re = (
    r"\b(?:"
    r"no|not|never|none|nothing|nowhere|neither|nor|"
    r"don['’']?t|doesn['’']?t|didn['’']?t|"
    r"can['’']?t|couldn['’']?t|won['’']?t|wouldn['’']?t|"
    r"shan['’']?t|shouldn['’']?t|mustn['’']?t|"
    r"isn['’']?t|aren['’']?t|wasn['’']?t|weren['’']?t|"
    r"ain['’']?t|hasn['’']?t|haven['’']?t|hadn['’']?t|"
    r"nah|nope|nuh[- ]uh|"
    r"hardly|scarcely|barely"
    r")\b"
)
    df["negation"] = text.str.contains(neg_re, regex=True, case=False)

    interj_re = (
        r"^(?:"
        r"ah+|aha+|ahh+|ahem|amen|aww+|aw|yay+|yaay+|woo+|wooo+|woah+|wow+|"
        r"hooray|hurrah|bravo|cheers|yippee|yahoo+|hurray|"
        r"oh+|ohh+|ooh+|oooh+|whoa+|whoops|oops|gosh|goodness|heavens|"
        r"jeez|gee+|holy|lordy|my+god|oh+dear|oh+my|"
        r"ugh+|ew+|yuck|ouch+|ow+|argh+|gah+|grr+|damn|darn|shoot|"
        r"bleh|brr+|pshh+|tsk+|sheesh|meh|bah|boo+|boohoo+|"
        r"uh+|uhh+|uhm+|umm+|um+|er+|erm+|hmm+|mhm+|hm+|"
        r"ok+|okay+|alright+|right+|indeed|sure+|"
        r"yeah+|yea+|yep+|yup+|yo+|okey+|roger|"
        r"hi+|hey+|hello+|hola+|ciao+|bye+|goodbye+|adieu|"
        r"psst+|hey+|yo+|look+|listen+|yohoo+|"
        r"whatever|seriously|really|pfft+|tch+|tss+|sheesh+|"
        r"mmm+|nom+|slurp+|sniff+|sigh+|huff+|"
        r"bah+|ben+|hein+|euh+|oula+|ohlala+|oh+la+la+|zut+|mince+|"
        r"super+|dommage+|tiens+|dis+donc+|"
        r"ouais+|nan+|non+|oui+|allez+|oula+"
        r")\b"
    )
    df["commence_par_interjection"] = (
        text.str.strip()
        .str.lower()
        .str.match(interj_re, case=False)
    )

    df["question_courte"]    = (df["nb_mots"] <= 5) & df["contient_point_interrogation"]
    df["exclamation_courte"] = (df["nb_mots"] <= 5) & df["contient_point_exclamation"]

    return df


def overall_feature_error_table(df, features):
    out = []
    for feat in features:
        present = df[df[feat]]
        absent  = df[~df[feat]]

        err_present = present["is_error"].sum()
        err_absent  = absent["is_error"].sum()

        er_p = err_present / len(present) if len(present) else 0.0
        er_a = err_absent / len(absent)  if len(absent)  else 0.0

        out.append({
            "caractéristique": feat,
            "effectif (présent)": len(present),
            "taux erreur (%) si présent": f"{round(er_p*100,2)} ({err_present}/{len(present)})" if len(present) else "0.0 (0/0)",
            "taux erreur (%) si absent": f"{round(er_a*100,2)} ({err_absent}/{len(absent)})" if len(absent) else "0.0 (0/0)",
            "rapport (présent/absent)": round((er_p/er_a), 3) if er_a > 0 else None
        })
    return pd.DataFrame(out).sort_values("taux erreur (%) si présent", ascending=False)


def per_class_feature_error_table(df, features, true_col):
    rows = []
    classes = sorted(df[true_col].unique())
    for c in classes:
        sub = df[df[true_col]==c]
        for feat in features:
            sub_p = sub[sub[feat]]
            sub_a = sub[~sub[feat]]

            err_present = sub_p["is_error"].sum()
            err_absent  = sub_a["is_error"].sum()

            er_p = err_present / len(sub_p) if len(sub_p) else 0.0
            er_a = err_absent / len(sub_a) if len(sub_a) else 0.0

            rows.append({
                "classe vraie": c,
                "caractéristique": feat,
                "effectif classe": len(sub),
                "effectif (présent)": len(sub_p),
                "taux erreur (%) si présent": f"{round(er_p*100,2)} ({err_present}/{len(sub_p)})" if len(sub_p) else "0.0 (0/0)",
                "taux erreur (%) si absent": f"{round(er_a*100,2)} ({err_absent}/{len(sub_a)})" if len(sub_a) else "0.0 (0/0)",
                "rapport (présent/absent)": round((er_p/er_a), 3) if er_a > 0 else None
            })
    return pd.DataFrame(rows)


def error_percentage_tables_from_preds(df, true_col, pred_col):
    labels = sorted(set(df[true_col]) | set(df[pred_col]))
    cm = pd.crosstab(
        pd.Series(df[true_col].values, name="classe vraie"),
        pd.Series(df[pred_col].values, name="prédit")
    ).reindex(index=labels, columns=labels, fill_value=0)

    support = cm.sum(axis=1)
    correct = np.diag(cm.values)
    errors = support - correct
    error_rate = (errors / support.replace(0, np.nan)).fillna(0) * 100

    pct_of_true = cm.div(support.replace(0, np.nan), axis=0) * 100
    for lab in labels:
        pct_of_true.loc[lab, lab] = 0.0
    errors_pct_of_true = pct_of_true.copy()
    errors_pct_of_true.insert(0, "taux erreur (%)", error_rate.round(2))

    pct_of_errors = cm.astype(float).copy()
    for i, lab in enumerate(labels):
        row_err = errors.iloc[i]
        if row_err > 0:
            pct_of_errors.iloc[i, :] = pct_of_errors.iloc[i, :] / row_err * 100.0
        else:
            pct_of_errors.iloc[i, :] = 0.0
        pct_of_errors.iloc[i, i] = 0.0
    errors_pct_of_errors = pct_of_errors.round(2)

    return errors_pct_of_true, errors_pct_of_errors, cm


def length_error_table(df):
    rows = []
    grouped = df.groupby("tranche_longueur", observed=False)
    for tranche, sub in grouped:
        err = sub["is_error"].sum()
        total = len(sub)
        taux = (err / total * 100) if total > 0 else 0.0
        rows.append({
            "tranche de longueur": tranche,
            "taux erreur (%)": f"{round(taux,2)} ({err}/{total})",
            "effectif": total
        })
    return pd.DataFrame(rows)


def per_class_length_error_table(df, true_col):
    rows = []
    grouped = df.groupby([true_col, "tranche_longueur"], observed=False)
    for (cls, tranche), sub in grouped:
        err = sub["is_error"].sum()
        total = len(sub)
        taux = (err / total * 100) if total > 0 else 0.0
        rows.append({
            "classe vraie": cls,
            "tranche de longueur": tranche,
            "taux erreur (%)": f"{round(taux,2)} ({err}/{total})",
            "effectif": total
        })
    return pd.DataFrame(rows)


def plot_global_errors_by_true_class(cm):
    support = cm.sum(axis=1).values
    correct = np.diag(cm.values)
    errors = support - correct
    labels = cm.index.tolist()

    mask = errors > 0
    if mask.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        errors[mask],
        labels=np.array(labels)[mask],
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel")
    )
    ax.set_title("Répartition globale des erreurs par classe vraie")
    return fig_to_base64(fig)


def plot_per_class_error_destinations(cm, max_classes=8):
    imgs = {}
    labels = cm.index.tolist()
    for i, true_lab in enumerate(labels):
        row = cm.iloc[i, :].astype(float).values
        err_row = row.copy()
        err_row[i] = 0.0
        total_err = err_row.sum()
        if total_err <= 0:
            continue

        idx_sorted = np.argsort(err_row)[::-1]
        vals_sorted = err_row[idx_sorted]
        labs_sorted = np.array(labels)[idx_sorted]

        if (vals_sorted > 0).sum() > max_classes:
            top_vals = vals_sorted[:max_classes]
            top_labs = labs_sorted[:max_classes]
            other_val = vals_sorted[max_classes:].sum()
            top_vals = np.append(top_vals, other_val)
            top_labs = np.append(top_labs, "autres")
        else:
            top_vals = vals_sorted[vals_sorted > 0]
            top_labs = labs_sorted[vals_sorted > 0]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            top_vals,
            labels=top_labs,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel")
        )
        ax.set_title(f"Répartition des erreurs de '{true_lab}'")
        imgs[true_lab] = fig_to_base64(fig)

    return imgs


def plot_feature_error_pies(df, features, true_col, pred_col):
    imgs = {}
    for feat in features:
        for cls in sorted(df[true_col].unique()):
            subset = df[(df[true_col] == cls) & (df["is_error"]) & (df[feat])]
            if subset.empty:
                continue
            counts = subset[pred_col].value_counts(normalize=True) * 100

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("pastel")
            )
            ax.set_title(f"{cls} | {feat}")
            imgs[f"{cls} | {feat}"] = fig_to_base64(fig)
    return imgs


def render_html(sections, pies_dicts):
    css = "<style>body{font-family:Arial} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px}</style>"
    html = ["<html><head>", css, "</head><body><h1>Analyse des erreurs</h1>"]

    for title, df in sections:
        html.append(f"<h2>{title}</h2>")
        html.append(df.to_html(index=False, escape=False))

    for title, imgs in pies_dicts:
        html.append(f"<h2>{title}</h2><div style='display:flex;flex-wrap:wrap'>")
        for key, img_b64 in imgs.items():
            html.append(f"<div style='margin:8px'><h4>{key}</h4><img src='{img_b64}'/></div>")
        html.append("</div>")

    html.append("</body></html>")
    return "\n".join(html)


def find_col(df, candidates):
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Fichier Excel coloré (entrée)")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie")
    parser.add_argument("--true_col", type=str, default="Emotion", help="Colonne true")
    parser.add_argument("--pred_col", type=str, default="predicted_emotion", help="Colonne prédite")
    parser.add_argument("--utt_col", type=str, default="Utterance", help="Colonne du texte")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.input)

    true_col = find_col(df, [args.true_col, "Emotion", "Label", "true_emotion"])
    if "predicted_emotion_final" in df.columns:
        pred_col = "predicted_emotion_final"
    else:
        pred_col = find_col(df, [args.pred_col, "predicted_emotion", "Prediction"])
    utt_col = find_col(df, [
        args.utt_col,
        "utterance", "Utterance", "text", "transcript", "utterance_id"
    ])


    if true_col is None or pred_col is None or utt_col is None:
        raise KeyError(
            f"[ERREUR] Impossible de trouver toutes les colonnes nécessaires. "
            f"Colonnes dispo: {df.columns.tolist()}"
        )

    before = len(df)
    df = df.dropna(subset=[true_col, pred_col, utt_col])
    print(f"[INFO] {before - len(df)} lignes supprimées (NaN)")

    df[true_col] = df[true_col].astype(str).str.strip().str.lower()
    df[pred_col] = df[pred_col].astype(str).str.strip().str.lower()
    df[utt_col] = df[utt_col].astype(str)

    df["is_error"] = df[true_col] != df[pred_col]
    df = build_features(df, utt_col)

    binary_features = [
        "contient_point_interrogation", "contient_point_exclamation",
        "negation", "commence_par_interjection",
        "question_courte", "exclamation_courte"
    ]

    global_table = overall_feature_error_table(df, binary_features)
    per_class_table = per_class_feature_error_table(df, binary_features, true_col)
    len_table = length_error_table(df)
    per_class_len = per_class_length_error_table(df, true_col)
    errors_pct_of_true, errors_pct_of_errors, cm = error_percentage_tables_from_preds(df, true_col, pred_col)

    global_pie = {"global": plot_global_errors_by_true_class(cm)} if cm is not None else {}
    per_class_pies = plot_per_class_error_destinations(cm)
    feature_pies = plot_feature_error_pies(df, binary_features, true_col, pred_col)

    sections = [
        ("Global — Taux d’erreur par caractéristique", global_table),
        ("Par classe — Taux d’erreur par caractéristique", per_class_table),
        ("Global — Taux d’erreur par tranche de longueur", len_table),
        ("Par classe — Taux d’erreur par tranche de longueur", per_class_len),
        ("Par classe — % erreurs relatives au support", errors_pct_of_true.reset_index()),
        ("Par classe — % distribution des erreurs", errors_pct_of_errors.reset_index()),
        ("Matrice de confusion", cm.reset_index())
    ]

    pies_dicts = [
        ("Camembert global", global_pie),
        ("Camemberts par classe", per_class_pies),
        ("Camemberts par feature+classe", feature_pies)
    ]

    html = render_html(sections, pies_dicts)
    out_html = os.path.join(args.output_dir, "rapport_erreurs.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Rapport HTML : {out_html}")


if __name__ == "__main__":
    main()
