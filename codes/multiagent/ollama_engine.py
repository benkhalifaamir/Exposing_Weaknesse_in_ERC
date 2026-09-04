from openai import OpenAI

class OllamaEngine:
    def __init__(self, base_url="http://localhost:11434/v1", api_key="ollama"):
        if "v1" not in base_url:
            base_url = f"{base_url}/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt, model="qwen2.5:72b", temperature=0.3, system_prompt=None):
        system_prompt = system_prompt or "You are a careful model that always returns strict JSON when asked."
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
