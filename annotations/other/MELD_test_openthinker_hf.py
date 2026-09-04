from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model from Hugging Face
model_name = "open-thoughts/OpenThinker-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use FP16 for better memory efficiency
    device_map="cuda"  # Move model to GPU
)

# Function to generate responses
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the model
prompt = "You are a specialist in MELD, a conversational dataset extracted from the TV series Friends. The dataset consists of dialogues involving multiple speakers, where each utterance is labeled with one of seven emotions. I want you to generate a conversation that follows the given emotion flow while maintaining the characteristics of MELD with speakers and their personalities and stay always in the same context. I want the response follows this structure without adding anything else : <Speaker> <text> <emotion> emotion flow : ['fear', 'neutral', 'neutral', 'sadness', 'neutral', 'joy']"
print(generate_response(prompt))
