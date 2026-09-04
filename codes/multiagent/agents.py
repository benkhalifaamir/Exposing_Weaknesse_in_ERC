import json
from ollama_engine import OllamaEngine

class DebateMemory:
    def __init__(self):
        self.history = []

    def add_turn(self, speaker, message, data=None):
        self.history.append({"speaker": speaker, "message": message, "data": data or {}})

class DebateAgent:
    def __init__(self, name, engine: OllamaEngine):
        self.name = name
        self.engine = engine
        self.memory = DebateMemory()

    def analyze(self, dialogue):
        prompt = f"""
You are {self.name}, an emotional reasoning agent.
Analyze this conversation and output strict JSON.

Focus on:
- polarity (positive/neutral/negative)
- speaker intention
- pragmatic/arousal cues
- transition likelihoods
- most probable emotion (Neutral, Joy, Sadness, Anger, Surprise, Fear, Disgust)

Conversation:
{dialogue}

Return JSON:
{{
  "analysis": [
    {{
      "utterance_id": <int>,
      "speaker": <string>,
      "utterance": <string>,
      "predicted_emotion": <string>,
      "polarity": <string>,
      "intention": <string>,
      "arousal": <string>,
      "transition_comment": <string>,
      "confidence": <float>
    }}
  ],
  "summary": <string>
}}
"""
        response = self.engine.generate(prompt)
        try:
            result = json.loads(response)
        except Exception:
            result = {"error": "Invalid JSON", "raw": response}
        self.memory.add_turn(self.name, "Analysis", result)
        return result

    def debate(self, other_agent, dialogue):
        prompt = f"""
You are {self.name}. Compare both analyses and produce a consensus emotion per utterance.

Return JSON:
{{
  "final_emotions": [
    {{
      "utterance_id": <int>,
      "final_emotion": <string>,
      "agreement": <bool>,
      "rationale": <string>
    }}
  ],
  "global_summary": <string>
}}

Agent A analysis: {json.dumps(self.memory.history[-1]['data'], ensure_ascii=False)}
Agent B analysis: {json.dumps(other_agent.memory.history[-1]['data'], ensure_ascii=False)}
"""
        response = self.engine.generate(prompt)
        try:
            result = json.loads(response)
        except Exception:
            result = {"error": "Invalid JSON", "raw": response}
        self.memory.add_turn(self.name, "Debate", result)
        return result
