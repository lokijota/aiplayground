# Simple langchain examples

All the samples call locally-running models on ollama.

- `basic_tutorial.py` - the simplest of all tutorials, based on https://python.langchain.com/docs/tutorials/llm_chain/ . Calling a model and using prompt templates.

- `docs_and_embeddings.py` - documents, document loaders, embedding models, storing embeddings and querying vector store (used chroma), retrievers. Based on https://python.langchain.com/docs/tutorials/retrievers/ .

- `text_classification.py` - tagging/classification of content. Includes structured outputs. Based on https://python.langchain.com/docs/tutorials/classification/ . With the local models I had partial success only, and also with portuguese text.