# generative-ai-zero-to-hero (work in progress)


## ML (Machine Learning) Fundamentals
- **Model** : Models are simplified representations/mathematical representation of real-world phenomena that enables explanations and can make predictions (it does so by identifying hidden patterns)
- **Features/Variables** : Input to model. They represent characteristics or properties of the data, and can be measured. In a patient medical dataset, features of patient could be age, gender, blood pressure, cholesterol level (https://domino.ai/data-science-dictionary/feature)
- **Weight(parameters)** : They represent the strength or importance of different features or variables in a model's prediction process. It dictates how model behaves and what results it produces
- **Array** : Collection of numbers. Model weights are represented as array
- **Vector** : 1 dimensional array (has magnitude and direction)

### Exercise
Consider the following models:
1. House price = 6500 * Number of sq ft + 2500000 (Purpose : House Price Prediction) 
2. Post Views = 0.4 * length of post + 2.1 * number of hashtags (Purpose : Social Media Post Popularity Prediction) 
3. Weekly grocery bill = 2300 * number of family members + 4500 (Purpose : Grocery bill estimation)

In the above models, what are the weights and what are the features/variables?

Some more key terms
- **Training** : The process of teaching a model to learn patterns and make predictions by exposing it to labeled data
- **Target Variable** : The variable the model aims to predict. For example if we want to predict the house price from size of the house, the house price is target variable
- **Independent variable** : The variables believed to influence/impact the target variable. For example if we want to predict the house price from size of the house, house size is an independent variable, as it impacts the house price.

The model weights are learnt by the model from data during the training process. Refer the Linear Regression Jupyter notebook for a simple example. The essence is we have real world data of 10 houses, which includes their area in square feet and their price. Based on that data, the model identifies some pattern/relationship between house price and house size. Using that identified pattern, if we are given the size of a new house, we can make a rough estimate of the house price even without asking the house owner about the actual price


## NLP (Natural Language Processing) Fundamentals


In a neural network, weights are the learned traits that determine the strength of a connection (or signal) between any two of the neurons that make up the content of the network. (https://blog.metaphysic.ai/weights-in-machine-learning/)



Parameter can include both model weights as well as hyperparameters



Token : A basic unit of text (for a language model). Tokens can be whole words, parts of words, punctuation marks, or even individual characters

Tokenization : The process of breaking down a piece of text into smaller, more manageable units called tokens. Visualize tokenization at https://platform.openai.com/tokenizer?WT.mc_id=academic-105485-koreyst


Representing text as vectors : Count based approach and Embedding based approach
Food for thought : Why represent text as vectors? (https://eavelardev.github.io/gcp_courses/nlp_on_gcp/text_representation/one_hot_encoding_and_bag_of_words.html)

Count based approach : One hot encoding, Bag of words, tfidf

(https://www.kaggle.com/code/vipulgandhi/bag-of-words-model-for-beginners)

Given this sentence, create a one hot encoded vector and bag of words vector
The dog chased the cat, and the cat chased the rat.

Use cases : 
1. Predict if a comment is toxic or not 
	- https://www.kaggle.com/code/faressayah/natural-language-processing-nlp-for-beginners
	- https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification
2. Predict if a mail is spam or not
3. Identify similar phrases in US patents to see if a newly filed patent already exists in patent database or not
	- https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/
4. Predict sentiment of a tweet on Twitter
	- https://www.kaggle.com/c/tweet-sentiment-extraction
5. Identify domain based on text in resume
6. Classify incoming documents and route to the right team

https://medium.com/@ebrahimhaqbhatti516/compilation-of-all-the-text-nlp-competitions-hosted-on-kaggle-17301835f225

Embedding based approach : Vectors are designed so that words with similar meanings have similar vector representations, capturing semantic and syntactic relationships between words

Prompt : the input a user provides to a model to guide it in producing a specific output
Prompt Engineering : Designing and optimizing prompts to deliver consistent and quality responses for a given application objective and model.

https://github.com/anthropics/prompt-eng-interactive-tutorial/tree/0d277542e927652da25b0014c9b346723af55881?tab=readme-ov-file

https://github.com/dair-ai/maven-pe-for-llms-4/tree/main

Prompt Engineering Design patterns

Primary Content Design pattern

We can improve prompt using
1. Examples (zero shot, one shot)
2. Cues : In this approach, we are giving the model a nudge in the right direction by starting it off with a snippet that reflects the desired response format.
3. Template


Training cutoff : Training cutoff refers to the date up to which the model's training data was collected and the model was trained (https://platform.openai.com/docs/models/o4-mini)

List of most popular LLMs : GPT-4o, o3, Gemini, Llama, R1, Claude, Qwen, Phi, Grok, Mistral
Open source : Llama 3, Deepseek R1, 

Context: The information the model uses to understand and generate text. Apart from input prompt, it can also include things like conversation history

Context Window : Refers to the limited amount of text a large language model (LLM) can consider at any given time when generating or understanding language. In simple words how much it can remember. It is measured in tokens

Function calling:  A mechanism where the LLM can identify the correct function to execute from a predefined set, based on the user's prompt, and determine the appropriate parameters to pass to that function

Chunking : Information is grouped into smaller, more manageable unitsRefer https://www.youtube.com/watch?v=8OJC21T2SL4

Information Retrieval : The process of finding and accessing relevant information from a large collection of text-based data (involves indexing, query processing, ranking, retrieval)

Structured data : Data follows structure i.e. organized in the form of row and columns

Unstructured data : Data which does not follow a predefined structure eg. audio, video, text

Semi structured data : Data which follows some structure, but not as rigid as structured data i.e. does not have the complete structure to fit into RDBMS eg. json, yaml, xml

Corpus : Large repository of documents/text

Query : What the use wants to search

Index : Pointer to the content of the database. Helps in quick and efficient retrieval of relevant document

Inverted Index : A data structure used primarily in text search engines. It maps terms (words) to their locations in a set of documents. This type of index allows for fast full-text searches, making it ideal for applications like search engines and document retrieval systems (https://www.pingcap.com/article/inverted-index-vs-other-indexes-key-differences/)

(https://www.youtube.com/watch?v=W4XSL_Osur4&list=PLz_RRnOnUTWETaBcpAMOd1evVIT3t8NSH&index=2)

Inverted Index/Posting list : Postings list is a data structure used in information retrieval to store the locations of terms within a document collection

Types of Inverted Indexes:
 - Record-level Inverted Index: Contains a list of document references for each term. 
 - Word-level Inverted Index: Additionally contains the positions of each word within a document. 

Exercise : Create an Inverted Index for the documents below (do it in Python)

Retrieval : Matching a query and a document. Basis of ranking algorithm that is used in search engine.

Query suggestion :

Query expansion : 

Refine intial query : 

Retrieval model : Include Boolean retrieval (document which contain exact words), Vector space (almost same as bag of words)

In essence, the VSM takes the BoW representation and translates it into a vector space where each document is a point in a high-dimensional space, with dimensions corresponding to the words in the vocabulary. 

https://stats.stackexchange.com/questions/31060/bag-of-words-vs-vector-space-model 

https://github.com/nouhadziri/Information-Retreival

What is a good retrieval algorithm?

Chroma db info retrieval : https://www.youtube.com/watch?v=gYhY-k4DQvE&list=PL58zEckBH8fA-R1ifTjTIjrdc3QKSk6hI
https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb
Reference: https://github.com/neo-con/chromadb-tutorial

RAG : https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb

Vector database: FAISS, ChromaDB, pgvector, Milvus, QDrant
https://python.langchain.com/docs/integrations/vectorstores/

Tool calling : Tool calling" and "function calling" are essentially the same concept, referring to the ability of a Large Language Model (LLM) to interact with external tools or functions to augment its capabilities beyond its own knowledge base.

Use cases of tool calling
	- Calling External Tools
	- Create API or Database Queries

https://gist.github.com/Donavan/62e238aa0a40ca88191255a070e356a2
https://www.reddit.com/r/LangChain/comments/1ifspwc/is_character_text_splitting_for_rag_still/?rdt=63486

Structured Chat : Structured chat LLM (or structured output LLM) is a Large Language Model (LLM) that is designed to produce responses in a specific, predictable, and organized format, often like JSON. This is in contrast to traditional LLMs that might produce unstructured text responses

Memory : 

Agent : LLM + Memory + Tools . Refer https://www.youtube.com/watch?v=1OLrT3dEzhA - build ai agent from scratch (https://github.com/neural-maze/agentic-patterns-course/blob/main/src/agentic_patterns/reflection_pattern/reflection_agent.py)
https://github.com/Ucas-HaoranWei/GOT-OCR2.0/tree/main?tab=readme-ov-file
Guardrails: 

Ollama: 
https://www.youtube.com/watch?v=1OLrT3dEzhA (build agent from scratch - The Neural Maze)
Small Language Models : 

https://zapier.com/blog/best-llm/

1. https://www.reddit.com/r/LocalLLM/comments/1e84286/intuitively_how_does_function_calling_work/
2. https://vtiya.medium.com/chucking-vs-tokenization-4805d8099885
3. https://www.reddit.com/r/LocalLLM/comments/1j1m4ep/14b_models_too_dumb_for_summarization/
4. https://www.youtube.com/watch?v=sVcwVQRHIc8 (build rag from scratch)
5. https://github.com/microsoft/generative-ai-for-beginners/tree/main/11-integrating-with-function-calling
6. https://www.reddit.com/r/MachineLearning/comments/1ax6j73/rag_vs_long_context_models_discussion/

1. Toeknization vs chunking?
2. Tool calling vs structured chat? Function calling helps give consistent response format, so does structured chat

3. With large context is rag required
For every concept, it is important that we develop the intuition. Some ways to develop intuition
1. Visualization (eg. visualize embeddings, tokenization)
2. Analogies
3. Build from scratch


Large Language Model
