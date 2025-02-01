import os
from langchain_openai import ChatOpenAI

# based in code from https://python.langchain.com/docs/tutorials/extraction/

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

model = ChatOpenAI(temperature=0, model="deepseek-r1", base_url = 'http://localhost:11434/v1')


from typing import Optional
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""
    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person, and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(default=None, description="The color of the person's hair if known")
    height_in_meters: Optional[str] = Field(default=None, description="Height measured in meters")


# Now let's create an information extractor

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

structured_model = model.with_structured_output(schema=Person)

text = "Donald Trump is 3 feet tall and has red hair."
prompt = prompt_template.invoke({"text": text})
result = structured_model.invoke(prompt)
print(result)

# Interesting. See what deepseek-r1 did there with the meters:
# name='Donald Trump' hair_color='red' height_in_meters="{convert(3, 'feet', 'meters')}"
# and llama3.2/Phi4/Gemma2:
# name='Donald Trump' hair_color='red' height_in_meters=None
