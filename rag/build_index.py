import pandas as pd
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

client = OpenAI(
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key=EMBEDDER_API_KEY
)


def chunk_text(text, max_len=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        start += max_len - overlap
    return chunks


def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",   # <=== если другая модель — поменяешь
        input=text
    )
    return resp.data[0].embedding


def main():
    df = pd.read_csv("./data/train_data.csv")
    all_chunks = []

    print("Чанкинг...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"])
        chunks = chunk_text(text)
        for ch in chunks:
            all_chunks.append({
                "doc_id": row["id"],
                "text": ch
            })

    print("Создаём эмбеддинги...")
    embeddings = []
    for item in tqdm(all_chunks):
        emb = get_embedding(item["text"])
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")

    print("Создаём FAISS индекс...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    print("Сохраняем...")
    faiss.write_index(index, "./data/faiss_index.bin")

    with open("./data/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Готово ✅")


if __name__ == "__main__":
    main()
