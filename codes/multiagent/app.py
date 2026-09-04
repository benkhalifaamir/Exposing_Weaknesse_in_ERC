from flask import Flask, render_template, request, jsonify
import pandas as pd
from ollama_engine import OllamaEngine
from agents import DebateAgent

app = Flask(__name__)

df = pd.read_csv("test_data.csv")
first_5 = df[df["Dialogue_ID"].isin(sorted(df["Dialogue_ID"].unique())[:5])]

engine = OllamaEngine()
agent1 = DebateAgent("Agent_A (Polarity/Intent)", engine)
agent2 = DebateAgent("Agent_B (Pragmatic/Transition)", engine)

@app.route("/")
def index():
    dialogues = []
    for did in first_5["Dialogue_ID"].unique():
        subset = first_5[first_5["Dialogue_ID"] == did]
        convo = [
            {"utterance_id": int(i), "speaker": row["Speaker"], "utterance": row["Utterance"]}
            for i, row in subset.iterrows()
        ]
        dialogues.append({"id": int(did), "dialogue": convo})
    return render_template("index.html", dialogues=dialogues)

@app.route("/debate_step", methods=["POST"])
def debate_step():
    payload = request.get_json()
    step = payload.get("step")
    dialogue = payload.get("dialogue")

    if step == "A":
        result = agent1.analyze(dialogue)
        return jsonify({"analysis": result})
    elif step == "B":
        result = agent2.analyze(dialogue)
        return jsonify({"analysis": result})
    elif step == "final":
        result = agent1.debate(agent2, dialogue)
        return jsonify({"result": result})
    else:
        return jsonify({"error": "Invalid step"}), 400

if __name__ == "__main__":
    app.run(debug=True)
