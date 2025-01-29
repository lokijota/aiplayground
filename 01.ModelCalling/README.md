# Simple model calling samples

All the samples call locally-running models on ollama.

- `manual_call.py` - use package requests and json to make http calls to openai-like interface.

- `openai_api_call.py` - use the openai package to do very similar calls to above. 

- `openai_api_structuredoutputs.py` - building on the previous example, but asking for a Json output and using Pydantic to parse it.

- `vision_ollama_api.py` - several examples of using a visual model to understand image content. Sometimes it seems amazingly accurate, others make silly mistakes (eg, miscounts tomatoes and says there's one inside the coffee mug). It also thinks that Innsbruck is where Munich is. But in general it's pretty good and also does OCR. Uses model llama3.2-vision.

- `vision_openai_api.py` - same calls as in the previous example but using the Openai API.

- `vision_openai_api_2images.py` - similar to the above but passing two images and asking a question about both. THIS EXAMPLE DOES NOT WORK, apparently llama3.2-vision only supports one image (or maybe ollama doesn't support two images). https://www.reddit.com/r/LocalLLaMA/comments/1h7lf9l/how_to_analyze_multiple_images_in_ollama_using/?rdt=49211 . 