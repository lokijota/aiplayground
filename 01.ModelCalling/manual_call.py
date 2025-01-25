import json
import requests


url = "http://localhost:11434/api/generate"

headers = { "Content-Type": "application/json"}

# https://github.com/ollama/ollama/blob/main/docs/api.md
data = {
    "model": "phi4",
    "prompt": "How do Munich and Lisbon compare? Pick one city to live! I know it's hard, but pick one and explain what gives it the edge.",
    "stream": False,
    "temperature": 1, # Temperature: Controls randomness, higher values increase diversity.
    "top_k": 50, # Top-k: Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens. Lower top-k also concentrates sampling on the highest probability tokens for each step.
    "top_p": 0.9 # Top-p (nucleus): The cumulative probability cutoff for token selection. Lower top-p values reduce diversity and focus on more probable tokens.
}
# num_ctx (context size?)

# https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542/2
# So temperature increases variety, while top-p and top-k reduce variety and focus samples on the model’s top predictions.
# You have to balance diversity and relevance when tuning these parameters for different applications.
# OpenAI recommends only altering either temperature or top-p from the default. Top-k is not exposed.

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    data = json.loads(response.text)
    actual_response = data["response"]
    print(actual_response)
    print("Total duration (s):", int(data["total_duration"])/1e9)
    print("Output tokens:", len(data["context"]))
else:
    print(f"Error: {response.status_code} - {response.text}")

# gemma2 chooses Lisbon after 19 seconds.
# deepseek-r1 chooses Lisbon after 58 seconds
# llama3.2 doesn't choose after 12 seconds, or it chose lisbon after increasing the temperature
# Phi4 doesn't chose after 48 seconds