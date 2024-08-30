# streamlit run <file location>/embeddingChatbot.py

# imports
import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
import numpy as np  # for normalizing embeddings
import streamlit as st

# models
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embeddings_path = "embedded_UWCourseCatalog_05-23-2024.csv"
df = pd.read_csv(embeddings_path)

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

# convert embeddings from CSV str type back to list type
df['ada_embedding'] = df['ada_embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    # cut dimension then normalize query embeddings in the same way as the dataset embeddings
    query_embedding = query_embedding[:256]
    query_embedding = normalize_l2(query_embedding)
    strings_and_relatednesses = [
        (row["combined"], relatedness_fn(query_embedding, row["ada_embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

# examples
# strings, relatednesses = strings_ranked_by_relatedness("public health", df, top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}")
#     print(string)

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the provided data on UW Madison college courses to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nClass Data:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about UW Madison college courses. Give some info about the course as well, such as the title, course code, and prerequisites."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

st.title("UW Madison Course Search")
# Initialize session state to keep track of the chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "query" not in st.session_state:
    st.session_state.query = ""

# User input: the textbox should be cleared after the user presses Enter
query = st.text_input("Enter keywords or ask a question:", value=st.session_state.query, key="query_input")

# Press Enter
if query:
    # Get response from the provided program
    response = ask(query)       
    # Add the query and response to the history
    st.session_state.history.append({"query": query, "response": response})
    st.session_state.query = ""

# Display the chat history, with the most recent query and response at the top
for item in reversed(st.session_state.history):
    st.write("---")
    st.write(f"Query:  {item['query']}")
    st.write(f"Response:  {item['response']}")