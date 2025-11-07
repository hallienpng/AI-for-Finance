import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

client = OpenAI(
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key=EMBEDDER_API_KEY
)

# Загружаем при первом импорте
faiss_index = faiss.read_index("./data/faiss_index.bin")

with open("./data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def retrieve(question, k=5):
    q_vec = get_embedding(question)
    D, I = faiss_index.search(np.array([q_vec]), k)
    retrieved = [chunks[i]["text"] for i in I[0]]
    return retrieved
