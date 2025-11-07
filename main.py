import os
import re
import json
import pickle
import random
import asyncio
import numpy as np
import pandas as pd
import faiss
import requests
import datetime
import tiktoken
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI, RateLimitError

load_dotenv()

# =====================================================
# MODELS + PRICES
# =====================================================

LLM_MODELS = {
    'mistral': 'openrouter/mistralai/mistral-small-3.2-24b-instruct',
    'llama': 'openrouter/meta-llama/llama-3-70b-instruct',
    'grok': 'openrouter/x-ai/grok-3-mini',
    'gemma': 'openrouter/google/gemma-3-27b-it',
}
EMBEDDER_MODELS = {
    'small': 'text-embedding-3-small',
    'ada': 'text-embedding-ada-002',
}

# Pricing per 1M tokens
PRICING_CATALOG = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "deepinfra/Qwen/Qwen3-Reranker-4B": {"input": 0.025, "output": 0.0},

    "openrouter/mistralai/mistral-small-3.2-24b-instruct": {"input": 0.06, "output": 0.18},
    "openrouter/meta-llama/llama-3-70b-instruct": {"input": 0.3, "output": 0.4},
    "openrouter/x-ai/grok-3-mini": {"input": 0.3, "output": 0.5},
    "openrouter/google/gemma-3-27b-it": {"input": 0.09, "output": 0.16},
}

ENCODER = tiktoken.get_encoding("cl100k_base")


# =====================================================
# Simple Cost Collector
# =====================================================
class CostCollector:
    def __init__(self):
        self.total_cost = 0.0

    def add(self, model_name, input_tokens, output_tokens):
        if model_name not in PRICING_CATALOG:
            print(f"⚠️ price missing for: {model_name}")
            return
        p = PRICING_CATALOG[model_name]
        cost = (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]
        self.total_cost += cost
        print(f"💰 Cost +{cost:.8f} USD (model={model_name}, input={input_tokens}, output={output_tokens})")

collector = CostCollector()


# =====================================================
# API KEYS
# =====================================================

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

if not EMBEDDER_API_KEY:
    raise ValueError("❌ EMBEDDER_API_KEY not found")
if not LLM_API_KEY:
    raise ValueError("❌ LLM_API_KEY not found")

EMBED_MODEL = EMBEDDER_MODELS["ada"]
LLM_MODEL = LLM_MODELS["grok"]
RERANK_MODEL = "deepinfra/Qwen/Qwen3-Reranker-4B"

INDEX_PATH = "train_data.index"
DOCS_PATH = "docs.pkl"


# =====================================================
# CLEAN MARKDOWN
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
# SAFE API CALLS
# =====================================================
async def safe_embed_call(client, texts, retries=5):
    input_tokens = sum(len(ENCODER.encode(t)) for t in texts)

    for attempt in range(1, retries + 1):
        try:
            r = await client.embeddings.create(
                model=EMBED_MODEL,
                input=texts
            )
            vectors = [x.embedding for x in r.data]

            collector.add(EMBED_MODEL, input_tokens, 0)
            return vectors

        except Exception as e:
            if attempt == retries:
                print(f"❌ EMBEDDING FAIL: {e}")
                return [np.zeros(1536).tolist()] * len(texts)

            wait = 2 + attempt * 2 + random.random() * 2
            print(f"⚠️ embed retry {attempt}/{retries}, wait {wait:.1f}s")
            await asyncio.sleep(wait)


async def safe_llm_call(client, prompt, retries=5):
    input_tokens = len(ENCODER.encode(prompt))

    for attempt in range(1, retries + 1):
        try:
            r = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                temperature=0.1
            )

            ans = r.choices[0].message.content
            output_tokens = len(ENCODER.encode(ans))

            collector.add(LLM_MODEL, input_tokens, output_tokens)
            return ans

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
# RERANK
# =====================================================
def rerank(query, docs, top_k=3):
    print("🔎 Reranking…")

    url = "https://ai-for-finance-hack.up.railway.app/rerank"

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDER_API_KEY}"
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])

        # cost
        tok_q = len(ENCODER.encode(query))
        tok_docs = sum(len(ENCODER.encode(x)) for x in docs)
        collector.add(RERANK_MODEL, tok_q + tok_docs, 0)

        results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        print(f"✅ Rerank done → top {top_k}")
        return results[:top_k]

    except Exception as e:
        print("❌ Rerank FAIL:", e)
        return []


# =====================================================
# BUILD INDEX
# =====================================================
async def build_index():
    print("📁 Loading train_data.csv")
    df = pd.read_csv("train_data.csv")

    print("✂️ Chunking...")
    chunks = []
    for _, row in df.iterrows():
        txt = clean_markdown_text(row["text"])
        if not txt:
            continue
        chunks.extend(split_financial_article(txt))

    print(f"✅ Total chunks = {len(chunks)}")

    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )

    print("⚡ Embedding chunks async…")
    vectors = []
    batch_size = 50

    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f" → Embedding batch {i//batch_size+1}/{(len(chunks)+batch_size-1)//batch_size}")
        tasks.append(safe_embed_call(embed_client, batch))

    results = await tqdm_asyncio.gather(*tasks)

    for r in results:
        vectors.extend(r)

    vectors = np.array(vectors).astype("float32")
    print(f"✅ Vectors shape = {vectors.shape}")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Index saved")


# =====================================================
# RETRIEVE
# =====================================================
async def retrieve(question, top_k_faiss=50, top_k_rerank=3):
    print(f"🔍 Retrieve for: {question[:60]}…")

    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )
    qvec = await safe_embed_call(embed_client, [question])
    qvec = np.array(qvec[0], dtype="float32").reshape(1, -1)

    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    print("   → FAISS search top50")
    D, I = index.search(qvec, top_k_faiss)
    top_docs = [docs[i] for i in I[0]]

    print("   → Rerank top3…")
    reranked = rerank(question, top_docs, top_k=top_k_rerank)

    if not reranked:
        print("⚠️ Rerank failed → using FAISS only")
        return top_docs[:top_k_rerank]

    final = []
    for r in reranked:
        idx = r["index"]
        final.append(top_docs[idx])

    print("✅ Retrieve done")
    return final


# =====================================================
# GENERATE ANSWER
# =====================================================
async def answer_generation(question):
    print(f"\n=========================\nQUESTION:\n{question}\n=========================")

    ctx = await retrieve(question)
    ctx_joined = "\n\n".join(ctx)

    print("🧠 Build prompt + call LLM…")

    prompt = f"""
Ты — высококвалифицированный финансовый помощник "AI for Finance Bank".
Отвечай строго на основе предоставленного контекста.
Если информации нет — ответь: "К сожалению, данной информации у меня нет…"

Контекст:
{ctx_joined}

Вопрос:
{question}

Ответ:
    """

    llm_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )

    ans = await safe_llm_call(llm_client, prompt)
    print("✅ LLM answer done")
    return ans


# =====================================================
# MAIN
# =====================================================
async def main():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        print("⚙️ No index → build…")
        await build_index()
    else:
        print("✅ Index found")

    df = pd.read_csv("questions.csv")
    ids = df["ID вопроса"].tolist()
    qs = df["Вопрос"].tolist()

    tasks = [answer_generation(q) for q in qs]
    answers = await tqdm_asyncio.gather(*tasks)

    out = pd.DataFrame({
        "ID вопроса": ids,
        "Вопрос": qs,
        "Ответы на вопрос": answers
    })

    out.to_csv("submission.csv", index=False)
    print("✅ submission.csv готов")
    print(f"\n💰 FINAL COST = {collector.total_cost:.6f} USD\n")


if __name__ == "__main__":
    asyncio.run(main())



