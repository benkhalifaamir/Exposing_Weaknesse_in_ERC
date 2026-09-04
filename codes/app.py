from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "super_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "analyse.xlsx")

ANNOTATION_COLUMNS = {
    "Intent_Clarity": ["Clear", "Unclear"],
    "Intent_Count": ["1", "2+"],
    "Lexical_Cue_Present": [
        "Yes",
        "None"
    ],
    "Punctuation": ["None","Weak","Strong"],
    "Context_Clarity": ["Clear", "Unclear"]
}

def load_data():
    df = pd.read_excel(DATA_FILE)

    for col in ANNOTATION_COLUMNS.keys():
        if col not in df.columns:
            df[col] = ""

    return df.fillna("")

@app.route("/")
def index():
    return redirect(url_for("annotate"))

@app.route("/annotate", methods=["GET"])
def annotate():
    df = load_data()
    data = df.to_dict("records")

    return render_template(
        "annotate.html",
        data=data,
        annotation_schema=ANNOTATION_COLUMNS,
        enumerate=enumerate
    )

@app.route("/save", methods=["POST"])
def save():
    df = load_data()

    for i in range(len(df)):
        for col in ANNOTATION_COLUMNS.keys():
            df.at[i, col] = request.form.get(f"{col}_{i}", "")

    df.to_excel(DATA_FILE, index=False)
    flash("Annotations saved successfully", "success")
    return redirect(url_for("annotate"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
