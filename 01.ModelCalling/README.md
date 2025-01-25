# Simple model calling samples

All the samples call locally-running models on ollama.

- `manual_call.py` - use package requests and json to make http calls to openai-like interface.

- `openai_api_call.py` - use the openai package to do very similar calls to above. 

- `openai_api_structuredoutputs.py` - building on the previous example, but asking for a Json output and using Pydantic to parse it.
