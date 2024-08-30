import pandas as pd
from openai import OpenAI
import tiktoken
import numpy as np

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

# from utils.embeddings_utils import get_embedding
embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000  # the maximum for text-embedding-3-small is 8191
client = OpenAI()

# load & inspect dataset
input_datapath = "UWCourseCatalog_05-23-2024.csv"
df = pd.read_csv(input_datapath)
print(df.columns)
df = df[["course-code","course-title","credits","description","mortarboard","Requisites:"]]
df = df.dropna()
df["combined"] = (
    "course-code: " + df['course-code'].astype(str).str.strip() + "; course-title: " + df['course-title'].astype(str).str.strip() + "; credits: " + df['credits'].astype(str).str.strip() + "; description: " + df['description'].astype(str).str.strip() + "; Requisites: " + df['Requisites:'].astype(str).str.strip()
)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

#cut dimension then normalize embeddings
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: normalize_l2(x[:256]))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace('      ', ' '))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace('     ', ' '))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace('    ', ' '))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace('   ', ' '))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace('  ', ' '))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace(' ', ', '))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: str(x).replace('[, ', '['))
#remove newlines
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: x.replace('\n', ''))

# use format(num, '.8f') to get rid of scientific notation and keep 8 decimal places]
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: x.strip('[]'))  # Remove brackets
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: ', '.join(filter(None, x.split(', '))))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: np.array(x.split(', '), dtype=float))
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: ', '.join(format(num, '.8f') for num in x))
# Add brackets back to the entire set of numbers
df['ada_embedding'] = df['ada_embedding'].apply(lambda x: f"[{x}]")

df.to_csv('embedded_UWCourseCatalog_05-23-2024.csv', index=False) 