import os
from langchain_openai import ChatOpenAI

# based in code from https://python.langchain.com/docs/tutorials/extraction/

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

model = ChatOpenAI(temperature=0, model="mistral-small", base_url = 'http://localhost:11434/v1')

# 01. Structured outputs

from typing import Optional, List
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
    # personality: Optional[str] = Field(default=None, description="Short summary of character traits of this person" )


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

text = "Marylin Monroe is 3 feet tall and has red hair."
prompt = prompt_template.invoke({"text": text})
result = structured_model.invoke(prompt)
print(result)

# Interesting. See what deepseek-r1 did there with the meters:
# name='Marylin Monroe' hair_color='red' height_in_meters="{convert(3, 'feet', 'meters')}"
# and llama3.2/Phi4/Gemma2/mistral-small:
# name='Marylin Monroe' hair_color='red' height_in_meters=None

text_maias = "Carlos Eduardo da Maia is one of the main characters in Eça de Queiroz's novel Os Maias. A lover of science and women. A dilettante, since he practises his profession for pleasure and not out of obligation. \
    He is the son of Pedro da Maia and Maria Monforte, but never had any contact with his parents, except when he was still a child. He is the main character in the novel. Many even consider Os Maias to be a character novel, centred precisely on Carlos da Maia. After Maria Monforte fled and his father committed suicide, he was left in the care of his grandfather Afonso da Maia, who gave him the education he couldn't give his son Pedro - brought up according to traditional Portuguese canons at the insistence of his ultra-Catholic mother - at the Douro estate of Santa Olávia, where he took refuge with his grandson, leaving Ramalhete abandoned. Thus, educated in the English way, with strict rules, intense physical activity, without the traditionalism of the Catholic ‘primer’ (which had tormented his father and poor Eusebiozinho, representatives of Portuguese education), Carlos would become a fine man, physically and intellectually. He graduates in Medicine in Coimbra, where he meets João da Ega, who becomes his great friend. \
    The house his grandfather had rented for him in Celas became the centre of bohemian student life, where art, politics and philosophy were discussed, making Carlos very popular with his classmates. After graduating, he travelled around Europe and got to know the best of the old continent. He becomes a dilettante. He returns to Lisbon and takes Afonso da Maia with him to Ramalhete. He works for pleasure, opens an office, sets up a laboratory and is full of projects that he never fulfils, scattered in the bohemian life of the capital, among women, friends and adventures. He maintains an adulterous relationship with the Countess of Gouvarinho, until he meets Maria Eduarda - who is actually his sister - with whom he falls in love. Unaware of the blood bond that unites them, they become lovers and decide first to run away together and, after discovering that Maria Eduarda was not in fact married, to wait for Afonso da Maia's death before getting married, until Carlos learns the terrible outcome of his story when he receives a letter from Mr Guimarães. Once the secret is out, Carlos Eduardo is haunted by his grandfather's death and becomes a failure in life. Thus, young, handsome, intelligent, coveted and cultured, with everything to become a winner, Carlos Eduardo da Maia is destined, like his father, Pedro, to fail. \
    Physically, Carlos da Maia was a handsome and magnificent young man, tall, well-built, with broad shoulders, black eyes, white skin, wavy black hair and a thin, dark brown beard, small and sharp on his chin. His moustache was curved at the corners of his mouth; psychologically he was cultured, well-educated, with refined tastes. He was brave and forthright, a friend to his friend and generous. His personality is characterised by cosmopolitanism, sensuality, a taste for luxury and dilettantism (someone who pursues their profession only for pleasure and not out of obligation). However, despite his education, Carlos failed, not because of it, but partly because of his surroundings - a parasitic, idle, futile and unstimulating society, and also because of hereditary aspects - his father's weakness and cowardice, his mother's selfishness, futility and bohemian spirit. Eça wanted to personify in Carlos the age of his youth, the age when he took part in the Coimbrã Question and the Casino Conferences and ended up in the group of Life's Losers, of which Carlos is a good example. He wore classic clothes, without many patterns. He was 1.74 meters high. "
prompt = prompt_template.invoke({"text": text_maias})
result = structured_model.invoke(prompt)
print(result)

# deepseek-r1 makes up the height based on text "tall"
# name='Carlos Eduardo da Maia' hair_color='wavy black' height_in_meters='1.85'
# while mistral-small:
# name='Carlos Eduardo da Maia' hair_color='black' height_in_meters=None
# interestingly, if I add "He was 3 meters high." at the end of the text, mistral says height = None. But if I say 2, it gets it right.
# If I say 1.5, it says None. If I say 1.74, gets it right. It's as if it's filtering out unrealistic values.


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

structured_model = model.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
result = structured_model.invoke(prompt)
print(result)


structured_model = model.with_structured_output(schema=Data)
prompt = prompt_template.invoke({"text": text_maias})
result = structured_model.invoke(prompt)

print(f"\nFound {len(result.people)}:")
for p in result.people:
    print(p)

# note: mistral-small finds 6 people in the text. But if I ask it to also describe the personality, only 4 are returned, and one of them has 
# personality None. Weird. This is all so undeterministic. Gemma finds 8 people (Countess of Gouvarinho and Mr Guimarães) but doesn't find Carlos' height. 


# Few shot prompting can be done passing more messages:
messages = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 🦜 4"},
]

response = model.invoke(messages)
print(response.content)

# mistral-small returns this (almost gets it):
# To determine what comes after 3 🦜 4, I need a bit more context. The sequence seems to involve numbers and an emoji (🦜), but the pattern isn't immediately clear.
# Could you please clarify the rule or pattern that applies to this sequence? For example:
# * Is it simply adding one to each number?
# * Is there some other mathematical operation involved?
# * Does the emoji have a specific role in the sequence?
# With more information, I can help you determine what comes next.
# gemma2 is able to deduce that the result is probably 7.



# 02. Tool calling

# This first part is weird, it seems to be about "teaching" the LLM about tools by provising few shot examples. From the tutorial text:

# "
# Structured output often uses tool calling under-the-hood. This typically involves the generation of AI messages containing tool calls,
# as well as tool messages containing the results of tool calls. What should a sequence of messages look like in this case?
#
# Different chat model providers impose different requirements for valid message sequences. Some will accept a (repeating) message
# sequence of the form:
#
# - User message
# - AI message with tool call
# - Tool message with result
# Others require a final AI message containing some sort of response.
#
# LangChain includes a utility function tool_example_to_messages that will generate a valid sequence for most model providers.
# It simplifies the generation of structured few-shot examples by just requiring Pydantic representations of the corresponding tool calls.
#
# Let's try this out. We can convert pairs of input strings and desired Pydantic objects to a sequence of messages that can be provided
# to a chat model. Under the hood, LangChain will format the tool calls to each provider's required format.
# "


from langchain_core.utils.function_calling import tool_example_to_messages

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]


messages = []

# LangChain includes a utility function tool_example_to_messages that will generate a valid sequence for most model providers.
# It simplifies the generation of structured few-shot examples by just requiring Pydantic representations of the corresponding tool calls.

for txt, tool_call in examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

for message in messages:
    message.pretty_print()



message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}

structured_llm = model.with_structured_output(schema=Data)
result = structured_llm.invoke([message_no_extraction])
print(result)
# people=[Person(name='John Doe', hair_color='brown', height_in_meters='1.75')]
# ^^ where the hell does this come from?! mistral totally hallucinates. But deepseek-r1 doesn't.
# people=[Person(name='Alice', hair_color='brown', height_in_meters=None), Person(name='Bob', hair_color='blonde', height_in_meters=None)]
# ^^ gemma2 also hallucinates badly, lol, what a disaster.

result =  structured_llm.invoke(messages + [message_no_extraction])
print(result)
# all models return people=[] here (correctl))

# one final test:
result =  structured_llm.invoke(messages + [text_maias])
print(result)