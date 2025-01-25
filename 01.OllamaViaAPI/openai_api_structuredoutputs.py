from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# based on https://github.com/ollama/ollama/blob/main/docs/openai.md , section "Structured outputs"

# Define the schema for the response
class CityInfo(BaseModel):
    name: str
    country: str 
    language: str
    population: int
    preferred: bool
    reasons: str

class CityList(BaseModel):
    cities: list[CityInfo]

# llama3.2, qwq, gemma2 handles structured outputs well
# phi4 couldn't fill in the population
# deepseek-r1 grossly exageraged munich's popoulation by 2x, otherwise it worked well (marked both as preferred...)

try:
    completion = client.beta.chat.completions.parse(
        temperature=0.3,
        model="gemma2",
        # model="llama3.2",
        messages=[
            {"role": "user", "content": "How do Munich and Lisbon compare? Pick one city to live! I know it's hard, but pick one and explain what gives it the edge. Return a list of cities in JSON format"}
        ],
        response_format=CityList,
    )

    cities_response = completion.choices[0].message
    if cities_response.parsed:
        for city in cities_response.parsed.cities:
            print(city, "\n")
    elif cities_response.refusal:
        print(cities_response.refusal)
except Exception as e:
    print(f"Error: {e}")