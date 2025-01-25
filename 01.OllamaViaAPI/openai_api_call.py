from openai import OpenAI

# https://ollama.com/blog/openai-compatibility

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="llama3.2",
  messages=[
    {"role": "system", "content": "You are a helpful life coach and finantial advisor."},
    {"role": "user", "content": "How do Munich and Lisbon compare? Pick one city to live! I know it's hard, but pick one and explain what gives it the edge."},
    # {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    # {"role": "user", "content": "Where was it played?"}
  ],
  temperature=0.7,
  top_p=0.5 # Top-p (nucleus): The cumulative probability cutoff for token selection. Lower top-p values reduce diversity and focus on more probable tokens.
)
# OpenAI recommends only altering either temperature or top-p from the default. Top-k is not exposed.

print(response.choices[0].message.content)
