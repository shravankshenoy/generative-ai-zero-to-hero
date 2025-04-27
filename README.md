# generative-ai-zero-to-hero (work in progress)

Model
Parameter
Vector
Representing text as vectors : Count based approach and Embedding based approach
Bag of words, tfidf
Embedding

Prompt : the input a user provides to a model to guide it in producing a specific output
Prompt Engineering : Designing and optimizing prompts to deliver consistent and quality responses for a given application objective and model.
Token : A basic unit of text (for a language model). Tokens can be whole words, parts of words, punctuation marks, or even individual characters
Tokenization : The process of breaking down a piece of text into smaller, more manageable units called tokens. Visualize tokenization at https://platform.openai.com/tokenizer?WT.mc_id=academic-105485-koreyst

Prompt Engineering Design patterns

Primary Content Design pattern

We can improve prompt using
1. Examples (zero shot, one shot)
2. Cues : In this approach, we are giving the model a nudge in the right direction by starting it off with a snippet that reflects the desired response format.
3. Template

Training : The process of teaching a model to learn patterns and make predictions by exposing it to labeled data

Training cutoff : Training cutoff refers to the date up to which the model's training data was collected and the model was trained (https://platform.openai.com/docs/models/o4-mini)

List of most popular LLMs : GPT-4o, o3, Gemini, Llama, R1, Claude, Qwen, Phi, Grok, Mistral
Open source : Llama 3, Deepseek R1, 

Context: The information the model uses to understand and generate text. Apart from input prompt, it can also include things like conversation history

Context Window : Refers to the limited amount of text a large language model (LLM) can consider at any given time when generating or understanding language. In simple words how much it can remember. It is measured in tokens

Function calling:  A mechanism where the LLM can identify the correct function to execute from a predefined set, based on the user's prompt, and determine the appropriate parameters to pass to that function

Chunking : Information is grouped into smaller, more manageable unitsRefer https://www.youtube.com/watch?v=8OJC21T2SL4

Information Retrieval : The process of finding and accessing relevant information from text-based data (involves indexing, query processing, ranking, retrieval)

RAG : 

Vector database: 

Tool calling : Tool calling" and "function calling" are essentially the same concept, referring to the ability of a Large Language Model (LLM) to interact with external tools or functions to augment its capabilities beyond its own knowledge base.

Structured Chat : Structured chat LLM (or structured output LLM) is a Large Language Model (LLM) that is designed to produce responses in a specific, predictable, and organized format, often like JSON. This is in contrast to traditional LLMs that might produce unstructured text responses

Memory : 

Agent : LLM + Memory + Tools . Refer https://www.youtube.com/watch?v=1OLrT3dEzhA - build ai agent from scratch (https://github.com/neural-maze/agentic-patterns-course/blob/main/src/agentic_patterns/reflection_pattern/reflection_agent.py)

Guardrails: 

Ollama: 

Small Language Models : 

https://zapier.com/blog/best-llm/

For every concept, it is important that we develop the intuition. Some ways to develop intuition
1. Visualization (eg. visualize embeddings, tokenization)
2. Analogies (connecting something new to something we already know)
3. Build from scratch

