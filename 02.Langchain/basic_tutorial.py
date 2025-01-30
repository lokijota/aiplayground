import os
from langchain_openai import ChatOpenAI

# mostly code from https://python.langchain.com/docs/tutorials/llm_chain/

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "ollama"

model = ChatOpenAI(model="llama3.2", base_url = 'http://localhost:11434/v1')

# 01. Let's first use the model directly an do some calls

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into informal european portuguese"),
    HumanMessage("hi!"),  # how are you and your wife and your dog?"),
]
# if I uncomment the text it just gives me a reply, it doesn't translate anything

response = model.invoke(messages) # type is AIMessage

print(response.content)
print(f"Prompt tokens: {response.usage_metadata['input_tokens']}, Output tokens:{response.usage_metadata['output_tokens']} ")

print(model.invoke("Banana").content, "\n-----")

print(model.invoke([{"role": "user", "content": "Hello"}]).content, "\n-----")

print(model.invoke([HumanMessage("This is not a pipe")]).content, "\n-----")

# 02. Use streaming to get progressive responses

for token in model.stream("Who wrote the Lusíadas? And tell me about its author."):
    print(token.content, end="|")

print()

# writes out several blocks of text, eg:

# The| L|us|í|adas| is| a| narrative| poem| written| by| Lu|ís| V|az| de| Cam|ões|,| a| Portuguese| poet|,| playwright|,| and| novelist|.| It| was| composed| between| |157|0| and| |158|0|.
# |Lu|ís| V|áz| de| Cam|ões| (|152|4|-|158|0|)| was| a| prominent| figure| in| European| literature| during| the| Renaissance| period|.| He| was| born| in| Le|ir|ia|,| Portugal|,| into| a| noble| family|.| Growing| up|,| he| studied| at| the| University| of| Co|im|bra|,| where| he| developed| an| interest| in| classical| Greek| and| Roman| cultures|.

# It may be just because of printing to the console, but each of these sentences show up at once and not totally streaming.
# Could also be a "framesize" config in the streaming.


# 03. Prompt Templates

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}" # {language} is a token to replace, duh.

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "European Portuguese", "text": "hey dude!"}) # note kvp/dictionary input

print(f"Prompt: \n", prompt) # list of messages
print(prompt.to_messages()) # list with a SystemMessage and a HumanMessage
# slighly different representation but the same content

response = model.invoke(prompt)
print(response.content)
