# Hugging Face Agents Course - Notes and Code
15/Feb/2025

## Unit 1 - Introduction to Agents
https://huggingface.co/learn/agents-course/unit1/introduction

Definition: An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.

Think of the Agent as having two main parts:

- The Brain (AI Model): This is where all the thinking happens. The AI model handles reasoning and planning. It decides which Actions to take based on the situation.

- The Body (Capabilities and Tools): This part represents everything the Agent is equipped to do. The scope of possible actions depends on what the agent has been equipped with. For example, because humans lack wings, they can’t perform the “fly” Action, but they can execute Actions like “walk”, “run” ,“jump”, “grab”, and so on.


**What type of tasks can an Agent do?**

An Agent can perform any task we implement via Tools to complete Actions.

For example, if I write an Agent to act as my personal assistant on my computer, and I ask it to “send an email to my Manager asking to delay today’s meeting”, I can give it some code to send emails. This will be a new Tool the Agent can use whenever it needs to send an email. We can write it in Python:

```
def send_message_to(recipient, message):
    """Useful to send an e-mail message to a recipient"""
    ...
```

The LLM, as we’ll see, will generate code to run the tool when it needs to, and thus fulfill the desired task.

```
send_message_to("Manager", "Can we postpone today's meeting?")
```

The design of the Tools is very important and has a great impact on the quality of your Agent. Some tasks will require very specific Tools to be crafted, while others may be solved with general purpose tools like “web_search”.

Note that *Actions* are not the same as *Tools*. An Action, for instance, can involve the use of multiple Tools to complete.

Allowing an agent to interact with its environment allows real-life usage for companies and individuals.

**Summary**

To summarize, an Agent is a system that uses an AI Model (typically an LLM) as its core reasoning engine, to:

- Understand natural language: Interpret and respond to human instructions in a meaningful way.
- Reason and plan: Analyze information, make decisions, and devise strategies to solve problems.
- Interact with its environment: Gather information, take actions, and observe the results of those actions.

### LLMs

There are 3 types of transformers :

1. Encoders: An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text.

    Example: BERT from Google
    - Use Cases: Text classification, semantic search, Named Entity Recognition
    - Typical Size: Millions of parameters

2. Decoders: A decoder-based Transformer focuses on generating new tokens to complete a sequence, one token at a time.

    Example: Llama from Meta
    - Use Cases: Text generation, chatbots, code generation
    - Typical Size: Billions (in the US sense, i.e., 10^9) of parameters

3. Seq2Seq (Encoder–Decoder): A sequence-to-sequence Transformer combines an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence.

    Example: T5, BART,
    - Use Cases: Translation, Summarization, Paraphrasing
    - Typical Size: Millions of parameters

Although Large Language Models come in various forms, **LLMs are typically decoder-based models with billions of parameters**. 


** Tokenizers **

The underlying principle of an LLM is simple yet highly effective: its objective is to predict the next token, given a sequence of previous tokens. A “token” is the unit of information an LLM works with. You can think of a “token” as if it was a “word”, but for efficiency reasons LLMs don’t use whole words.

For example, while English has an estimated 600,000 words, an LLM might have a vocabulary of around 32,000 tokens (as is the case with Llama 2). Tokenization often works on sub-word units that can be combined.

Each LLM has some special tokens specific to the model. The LLM uses these tokens to open and close the structured components of its generation. For example, to indicate the start or end of a sequence, message, or response. Moreover, the input prompts that we pass to the model are also structured with special tokens. The most important of those is the End of sequence token (EOS).


| Model | Provider | EOS Token | Functionality
| --- | --- | --- | --- | 
| GPT4	| OpenAI | `<\|endoftext\|>`	| End of message text
| Llama 3	| Meta (Facebook AI Research)	| `<\|eot_id\|>`	| End of sequence
| Deepseek-R1 | DeepSeek	| `<\|end_of_sentence\|>`	| End of message text
| SmolLM2 | Hugging Face | `<\|im_end\|>`	| End of instruction or message
| Gemma	| Google	| `<end_of_turn>` | End of conversation turn


#### Understanding next token prediction.

LLMs are said to be *autoregressive*, meaning that the output from one pass becomes the input for the next one. This loop continues until the model predicts the next token to be the EOS token, at which point the model can stop.

In other words, an LLM will decode text until it reaches the EOS. But what happens during a single decoding loop?

While the full process can be quite technical, here’s a brief overview:

- Once the input text is tokenized, the model computes a representation of the sequence that captures information about the meaning and the position of each token in the input sequence.
- This representation goes into the [decoder] model, which outputs scores that rank the likelihood of each token in its vocabulary as being the next one in the sequence.

Based on these scores, we have multiple strategies to select the tokens to complete the sentence.
- The easiest decoding strategy would be to always take the token with the maximum score.

*(notajota: there's a nice space that shows the generation of next words and picking of one. In the case it always picks the top one, but as you know the temperature and top-k control this behaviour)*

But there are more advanced decoding strategies. For example, *beam search* explores multiple candidate sequences to find the one with the maximum total score–even if some individual tokens have lower scores.

*(notajota: nice simulator again for the beam search visualizer, in-page, here: https://huggingface.co/learn/agents-course/unit1/what-are-llms)*

#### Attention is all you need

A key aspect of the Transformer architecture is Attention. When predicting the next word, not every word in a sentence is equally important; words like Portugal and “capital” in the sentence “The capital of Portugal is …” carry the most meaning.

This process of identifying the most relevant words to predict the next token has proven to be incredibly effective.

Although the basic principle of LLMs—predicting the next token—has remained consistent since GPT-2, there have been significant advancements in scaling neural networks and making the attention mechanism work for longer and longer sequences.

If you’ve interacted with LLMs, you’re probably familiar with the term *context length*, which refers to the maximum number of tokens the LLM can process, and the maximum attention span it has.

- To generate HF tokens: https://huggingface.co/settings/tokens
- To ask for access to Llama: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### Messages and Special Tokens - https://huggingface.co/learn/agents-course/unit1/messages-and-special-tokens

System messages (also called System Prompts) define how the model should behave. They serve as persistent instructions, guiding every subsequent interaction:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

When using Agents, the System Message also gives information about the available tools, provides instructions to the model on how to format the actions to take, and includes guidelines on how the thought process should be segmented.


A conversation consists of alternating messages between a Human (user) and an LLM (assistant).

*Chat templates* help maintain context by preserving conversation history, storing previous exchanges between the user and the assistant. This leads to more coherent multi-turn conversations.

For example:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

This conversation would be translated into the following prompt when using Llama 3.2:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 10 Feb 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

I need help with my order<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'd be happy to help. Could you provide your order number?<|eot_id|><|start_header_id|>user<|end_header_id|>

It's ORDER-123<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Templates can handle complex multi-turn conversations while maintaining context:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

#### Chat teamplates, Base Models vs. Instruct Models

Another point we need to understand is the difference between a Base Model vs. an Instruct Model:

A *Base Model* is trained on raw text data to predict the next token.

An *Instruct Model* is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in.

ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant). If you have interacted with some AI API lately, you know that’s the standard practice.

It’s important to note that a base model could be fine-tuned on different chat templates, so when we’re using an instruct model we need to make sure we’re using the correct chat template.

*(Notajota: there's some explanation here about chat templates, to transform sequences of prompts/responses - assistant/user into a format with separators that is then used to do instruction-tuning)*

The transformers library will take care of chat templates for you as part of the tokenization process. Read more about how transformers uses chat templates here. All we have to do is structure our messages in the correct way and the tokenizer will take care of the rest.

#### From Messages to prompt

The easiest way to ensure your LLM receives a conversation correctly formatted is to use the chat_template from the model’s tokenizer.


```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
```

To convert the previous conversation into a prompt, we load the tokenizer and call apply_chat_template:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The `rendered_prompt` returned by this function is now ready to use as the input for the model you chose!

### What are Tools - https://huggingface.co/learn/agents-course/unit1/tools

One crucial aspect of AI Agents is their ability to take actions. As we saw, this happens through the use of Tools. By giving your Agent the right Tools — and clearly describing how those Tools work — you can dramatically increase what your AI can accomplish. 

A Tool is a function given to the LLM. This function should fulfill a clear objective. A good tool should be something that complements the power of an LLM.

Furthermore, LLMs predict the completion of a prompt based on their training data, which means that their internal knowledge only includes events prior to their training. Therefore, if your agent needs up-to-date data you must provide it through some tool.

A Tool should contain:
- A textual description of what the function does.
- A Callable (something to perform an action).
- Arguments with typings.
- (Optional) Outputs with typings.

A good tool should be something that complements the power of an LLM. For instance, if you need to perform arithmetic, giving a calculator tool to your LLM will provide better results than relying on the native capabilities of the model.

Furthermore, LLMs predict the completion of a prompt based on their training data, which means that their internal knowledge only includes events prior to their training. Therefore, if your agent needs up-to-date data you must provide it through some tool.

#### How do tools work?

LLMs, as we saw, can only receive text inputs and generate text outputs. They have no way to call tools on their own. **What we mean when we talk about providing tools to an Agent, is that we teach the LLM about the existence of tools, and ask the model to generate text that will invoke tools when it needs to.** For example, if we provide a tool to check the weather at a location from the Internet, and then ask the LLM about the weather in Lisbon, the LLM will recognize that question as a relevant opportunity to use the “weather” tool we taught it about. The LLM will generate text, in the form of code, to invoke that tool. **It is the responsibility of the Agent to parse the LLM’s output, recognize that a tool call is required, and invoke the tool on the LLM’s behalf. The output from the tool will then be sent back to the LLM, which will compose its final response for the user.**

The output from a tool call is another type of message in the conversation. *Tool calling steps are typically not shown to the user: the Agent retrieves the conversation, calls the tool(s), gets the outputs, adds them as a new conversation message, and sends the updated conversation to the LLM again.* From the user’s point of view, it’s like the LLM had used the tool, but in fact it was our application code (the Agent) who did it.

#### How do we give tools to an LLM?

The complete answer may seem overwhelming, but we essentially use the system prompt to provide textual descriptions of available tools to the model.

If this seems too theoretical, let’s understand it through a concrete example.  We will implement a simplified calculator tool that will just multiply two integers. This could be our Python implementation:

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

All of the details of the interface are important. Let’s put them together in a text string that describes our tool for the LLM to understand:

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

When we pass the previous string as part of the input to the LLM, the model will recognize it as a tool, and will know what it needs to pass as inputs and what to expect from the output. If we want to provide additional tools, we must be consistent and always use the same format. This process can be fragile, and we might accidentally overlook some details. Is there a better way?

#### Auto-formatting Tool sections

Our tool was written in Python, and the implementation already provides everything we need:

We will leverage Python’s introspection features to leverage the source code and build a tool description automatically for us. All we need is that the tool implementation uses type hints, docstrings, and sensible function names. We will write some code to extract the relevant portions from the source code.

After we are done, we’ll only need to use a Python decorator to indicate that the calculator function is a tool:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

Note the `@tool` decorator before the function definition.

So that is this Tool thingie?

#### Generic Tool implementation

*Disclaimer: This example implementation is fictional but closely resembles real implementations in most libraries.*

```python
class Tool:
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 func: callable, 
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool, 
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
```

We could create a Tool with this class using code like the following:

```python
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)
```

But we can also use Python’s inspect module to retrieve all the information for us! This is what the @tool decorator does.

This is not related to the course - to implement a decorator in python:

```python
def tool(func):
    """
    A decorator that creates a Tool instance from the given function.
    """
    # Get the function signature
    signature = inspect.signature(func)
    
    # Extract (param_name, param_annotation) pairs for inputs
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__ 
            if hasattr(param.annotation, '__name__') 
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))
    
    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__ 
            if hasattr(return_annotation, '__name__') 
            else str(return_annotation)
        )
    
    # Use the function's docstring as the description (default if None)
    description = func.__doc__ or "No description provided."
    
    # The function name becomes the Tool name
    name = func.__name__
    
    # Return a new Tool instance
    return Tool(
        name=name, 
        description=description, 
        func=func, 
        arguments=arguments, 
        outputs=outputs
    )
```

Now we can use the Tool’s ```to_string``` method to automatically retrieve a text suitable to be used as a tool description for an LLM. The description is then injected in the system prompt.


[Continue (this page is too long already)](README2.md)