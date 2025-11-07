import os
import re
import time
import json
import pickle
import random
import asyncio
import numpy as np
import pandas as pd
import faiss
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI, RateLimitError

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

if not EMBEDDER_API_KEY:
    raise ValueError("❌ EMBEDDER_API_KEY not found")

if not LLM_API_KEY:
    raise ValueError("❌ LLM_API_KEY not found")

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "openrouter/mistralai/mistral-small-3.2-24b-instruct"

INDEX_PATH = "train_data.index"
DOCS_PATH = "docs.pkl"


# =====================================================
# MARKDOWN CLEANING
# =====================================================
def clean_markdown_text(text: str):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    text = re.sub(r'>{1,}\s*', '', text)
    text = re.sub(r'[-*+]\s*', '', text)
    text = re.sub(r'\d+\.\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# =====================================================
# SAFE ASYNC REQUESTS
# =====================================================
async def safe_embed_call(client, texts, retries=5):
    for attempt in range(1, retries + 1):
        try:
            r = await client.embeddings.create(
                model=EMBED_MODEL,
                input=texts
            )
            return [x.embedding for x in r.data]

        except Exception as e:
            if attempt == retries:
                print(f"❌ EMBEDDING FAIL: {e}")
                return [np.zeros(1536).tolist()] * len(texts)

            wait = 2 + attempt * 2 + random.random() * 2
            print(f"⚠️ embed retry {attempt}/{retries}, wait {wait:.1f}s")
            await asyncio.sleep(wait)


async def safe_llm_call(client, prompt, retries=5):
    for attempt in range(1, retries + 1):
        try:
            r = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                temperature=0.1
            )
            return r.choices[0].message.content

        except RateLimitError:
            wait = 5 + attempt * 3 + random.random() * 2
            print(f"⚠️ RateLimit {attempt}/{retries}, wait {wait:.1f}s")
            await asyncio.sleep(wait)

        except Exception as e:
            if attempt == retries:
                print(f"❌ LLM FAIL: {e}")
                return "Ошибка при генерации ответа."
            wait = 2 + attempt * 2
            print(f"⚠️ LLM retry {attempt}/{retries}, wait {wait:.1f}s")
            await asyncio.sleep(wait)


# =====================================================
# CHUNKING
# =====================================================
def split_by_sections(text):
    return [s for s in text.split("\n\n") if len(s.strip()) > 100]


def split_long_section(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c.strip()) > 100]


def split_financial_article(text):
    chunks = []
    for sec in split_by_sections(text):
        if len(sec) <= 800:
            chunks.append(sec)
        else:
            chunks.extend(split_long_section(sec))
    return chunks


# =====================================================
# INDEX BUILDING
# =====================================================
async def build_index():
    print("📁 Загрузка train_data.csv")
    df = pd.read_csv("train_data.csv")

    chunks = []
    print("✂️ чанкинг...")
    for _, row in df.iterrows():
        txt = clean_markdown_text(row["text"])
        if not txt:
            continue
        chunks.extend(split_financial_article(txt))

    print(f"✅ total chunks: {len(chunks)}")

    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )

    print("⚡ async embedding...")
    vectors = []
    batch_size = 50

    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks.append(safe_embed_call(embed_client, batch))

    results = await tqdm_asyncio.gather(*tasks)

    for r in results:
        vectors.extend(r)

    vectors = np.array(vectors).astype("float32")

    print(f"✅ vectors: {vectors.shape}")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ index saved")


# =====================================================
# RETRIEVAL
# =====================================================
async def retrieve(question, k=3):
    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )

    qvec = await safe_embed_call(embed_client, [question])
    qvec = np.array(qvec[0], dtype="float32").reshape(1, -1)

    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    D, I = index.search(qvec, k)
    return [docs[i] for i in I[0]]


# =====================================================
# ANSWERING
# =====================================================
async def answer_generation(question):
    ctx = await retrieve(question, k=3)
    ctx = "\n\n".join(ctx)

    prompt = f"""
Ты — высококвалифицированный финансовый помощник, общающийся с клиентом.
Твоя задача - дать точный и полный ответ на вопрос клиента, используя только предоставленные ниже статьи из базы знаний.
Не придумывай ничего, чего нет в тексте.
Если в статьях нет ответа на вопрос, вежливо сообщи: "К сожалению, я не могу ответить на данный вопрос."
Отвечай на русском языке.

Контекст:
{ctx}

Вопрос:
{question}

Ответ:
    """

    llm_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )

    return await safe_llm_call(llm_client, prompt)


# =====================================================
# MAIN
# =====================================================
async def main():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        print("⚙️ Индекса нет → создаём")
        await build_index()
    else:
        print("✅ index found")

    df = pd.read_csv("questions.csv")
    qs = df["Вопрос"].tolist()

    tasks = [answer_generation(q) for q in qs]
    answers = await tqdm_asyncio.gather(*tasks)

    df["Ответы на вопрос"] = answers
    df.to_csv("submission.csv", index=False)
    print("✅ submission.csv готов")


if __name__ == "__main__":
    asyncio.run(main())

