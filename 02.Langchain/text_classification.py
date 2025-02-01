import os
from langchain_openai import ChatOpenAI

# based in code from https://python.langchain.com/docs/tutorials/classification/

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

# Let's specify a Pydantic data model with a few properties and their expected type in our schema

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(description="How aggressive or violent the text is on a scale from 1 to 10")
    language: str = Field(description="The ISO code of the language the text is written in")

model = ChatOpenAI(temperature=0, model="deepseek-r1", base_url = 'http://localhost:11434/v1').with_structured_output(Classification)

sentences = [ "És uma pessoa horrível e odeio-te e vou matar-te com as minhas próprias mãos!", \
              "Dâmaso balbuciava, escarlate: - Ora essa, minha senhora! O que lhe ﬁz?... Carícias, sempre carícias...", \
              "You are an horrible person and I hate you and I'm going to kill you with my bear (yes bear) hands!", \
              "És um doce.", \
              "You are sweet.", \
              "Ésa es natural condición de mujeres -dijo don Quijote-: desdeñar a quien las quiere y amar a quien las aborrece. Pasa adelante, Sancho." ]

for sentence in sentences:
    prompt = tagging_prompt.invoke({"input": sentence})
    response = model.invoke(prompt)
    print(response)
    print(response.model_dump()) # dictionary output

# I tried all the models I have locally on ollama. All got bad and inconsistent results. deepseek-r1 was the least worse, but still with several problems.
# sentiment='negative' aggressiveness=1 language='pt'
# sentiment='negative' aggressiveness=0 language='Portuguese'
# sentiment='negative' aggressiveness=10 language='en'
# sentiment='negative' aggressiveness=0 language='Portuguese'
# sentiment='positive' aggressiveness=0 language='en'
# sentiment='negative' aggressiveness=0 language='es'

# trying with finer control

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment or emotion of the text", enum=["happy", "neutral", "sad", "angry", "frustrated", "sweet"])
    aggressiveness: int = Field(
        description="describes how aggressive or violent the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(description="The ISO code of the language the text is written in", enum=["es", "en", "fr", "de", "pt"])

model = ChatOpenAI(temperature=0, model="deepseek-r1", base_url = 'http://localhost:11434/v1').with_structured_output(Classification)

# adding verbose output
import langchain
langchain.debug = True

print("--- Finer control prompts ---")
for sentence in sentences:
    prompt = tagging_prompt.invoke({"input": sentence})
    response = model.invoke(prompt)
    print(response)

# at least the language codes are standardized, but why fr? and agressiveness always 1, and several sentiments wrong
# sentiment='neutral' aggressiveness=1 language='pt'
# sentiment='neutral' aggressiveness=1 language='pt'
# sentiment='happy' aggressiveness=1 language='en'
# sentiment='neutral' aggressiveness=1 language='pt'
# sentiment='neutral' aggressiveness=1 language='fr'
# sentiment='neutral' aggressiveness=1 language='es'

# Note: the langchain.debug = True does add a lot of debug, but doesn't show how the Classification block is added to the prompt
# another alternative for this is to use callbacks: https://python.langchain.com/docs/how_to/custom_callbacks/ (of perhaps mlflow)


# Later on I found this: https://github.com/NihilDigit/DeepSeek-Structured-Output-for-LangChain , leaving it here as reference:
# When using DeepSeek models with LangChain's ChatOpenAI, the built-in with_structured_output() method is not supported.
# This implementation provides a solution by extending the ChatOpenAI class with DeepSeek-compatible structured output functionality.