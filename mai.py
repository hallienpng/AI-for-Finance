import os
import re
import json
import pickle
import random
import asyncio
import requests
import numpy as np
import pandas as pd
import faiss
import gc
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
KEYWORDS_CACHE_PATH = "keywords_cache.json"
ANSWERS_CACHE_PATH = "submission_cache.json"

# ==============================
# MARKDOWN CLEANING
# ==============================
def clean_markdown_text(text: str):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ==============================
# SAFE ASYNC REQUESTS
# ==============================
async def safe_embed_call(client, texts, retries=5):
    for attempt in range(1, retries + 1):
        try:
            r = await client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [x.embedding for x in r.data]
        except Exception as e:
            if attempt == retries:
                print(f"❌ EMBEDDING FAIL: {e}")
                return None
            wait = 2 + attempt * 2 + random.random() * 2
            print(f"⚠️ embed retry {attempt}/{retries}, wait {wait:.1f}s")
            await asyncio.sleep(wait)

async def safe_llm_call(client, prompt, retries=5):
    for attempt in range(1, retries + 1):
        try:
            r = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
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

# ==============================
# SECTION PARSING
# ==============================
def parse_sections(doc_id, text, annotation, tags):
    sections = []
    parts = re.split(r'\n##\s*', text)
    prefix = parts[0].strip()
    idx = 0
    if prefix:
        sections.append({
            "doc_id": doc_id,
            "section_title": "intro",
            "section_idx": idx,
            "section_text": prefix,
            "annotation": annotation,
            "tags": tags
        })
        idx += 1
    for p in parts[1:]:
        lines = p.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections.append({
            "doc_id": doc_id,
            "section_title": title,
            "section_idx": idx,
            "section_text": body,
            "annotation": annotation,
            "tags": tags
        })
        idx += 1
    return sections

# ==============================
# CHUNKING
# ==============================
def chunk_section(section, chunk_size=800, overlap=80, min_len=100):
    text = section["section_text"]
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    parts = splitter.split_text(text)
    chunk_idx = 0
    for part in parts:
        t = part.strip()
        if len(t) < min_len:
            continue
        chunks.append({
            "doc_id": section["doc_id"],
            "section_title": section["section_title"],
            "section_idx": section["section_idx"],
            "chunk_idx": chunk_idx,
            "text": t,
            "annotation": section["annotation"],
            "tags": section["tags"]
        })
        chunk_idx += 1
    return chunks

# ==============================
# STOPWORDS
# ==============================
RUS_STOPWORDS = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
    "его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было",
    "вот","от","меня","еще","нет","о","из","ему","теперь","когда","даже","ну","вдруг",
    "ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
    "уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где",
    "есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто",
    "чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому"
}

# ==============================
# REGEXP + LLM KEYWORD EXTRACTION WITH CACHE
# ==============================
LLM_CONCURRENCY_LIMIT = 2
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)

if os.path.exists(KEYWORDS_CACHE_PATH):
    with open(KEYWORDS_CACHE_PATH, "r", encoding="utf-8") as f:
        LLM_KEYWORD_CACHE = json.load(f)
else:
    LLM_KEYWORD_CACHE = {}

def save_keywords_cache():
    with open(KEYWORDS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(LLM_KEYWORD_CACHE, f, ensure_ascii=False, indent=2)

def regexp_extract_keywords(question: str, min_len=3):
    text = question.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)
    lemmas = [t for t in tokens if len(t) >= min_len and t not in RUS_STOPWORDS]
    bigrams = [lemmas[i] + " " + lemmas[i + 1] for i in range(len(lemmas)-1)]
    return list(set(lemmas + bigrams))

async def llm_extract_keywords(question: str, llm_client):
    if question in LLM_KEYWORD_CACHE:
        return LLM_KEYWORD_CACHE[question]
    prompt = f"""
Ты — помощник, который извлекает ключевые слова и ключевые фразы из заданного вопроса.
Выход должен быть в формате JSON: {{ "keywords": ["слово1", "фраза2", ...] }}

Особенности:
- Убирай стоп-слова.
- Выводи как отдельные ключевые слова, так и словосочетания.
- Слова могут быть разного рода/склонения. Например, "мошеннические" → ключевое слово "мошенничество".
- Выводи только ключевые понятия, релевантные теме вопроса.

Вопрос: "{question}"
""".strip()
    async with llm_semaphore:
        try:
            resp = await safe_llm_call(llm_client, prompt)
            data = json.loads(resp)
            kws = data.get("keywords", [])
        except Exception as e:
            print(f"❌ Ошибка LLM при получении ключевых слов: {e}")
            kws = []

        LLM_KEYWORD_CACHE[question] = kws
        save_keywords_cache()
        return kws

async def extract_query_keywords(question: str, llm_client=None, min_len=3):
    kws = regexp_extract_keywords(question, min_len=min_len)
    if len(kws) < 3 and llm_client is not None:
        kws_llm = await llm_extract_keywords(question, llm_client)
        kws = list(set(kws + kws_llm))
    return kws

# ==============================
# COUNT TAG MATCHES
# ==============================
def count_tag_matches(question_tokens, tags):
    if not tags:
        return 0
    score = 0
    tags_norm = []
    for t in tags:
        t = t.lower()
        t = re.sub(r"[^a-zа-яё0-9]+", " ", t)
        tags_norm.extend(t.split())
    for tok in question_tokens:
        if tok in tags_norm:
            score += 1
    return score

# ==============================
# RERANK
# ==============================
def rerank_docs(query, documents, key):
    url = "https://ai-for-finance-hack.up.railway.app/rerank"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": "deepinfra/Qwen/Qwen3-Reranker-4B", "query": query, "documents": documents}
    resp = requests.post(url, headers=headers, json=payload)
    try:
        data = resp.json()
        return data
    except Exception:
        print("❌ error in rerank_docs response:", resp.text)
        return {"results": []}

# ==============================
# RETRIEVE
# ==============================
async def retrieve(question, k=3, prefilter_top=50, faiss_top=30):
    print("\n=== RETRIEVE ===")
    print(f"[+] question: {question}")

    llm_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )

    kw = await extract_query_keywords(question, llm_client)
    print(f"[+] keywords: {kw}")

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    # ===== FILTER BY TAGS =====
    candidate_docs = []
    for doc in docs:
        tags = doc.get("tags", [])
        score = 0
        for t in tags:
            t_norm = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", t.lower())
            for tt in t_norm:
                if tt in kw:
                    score += 2
                elif any(tt in term for term in kw):
                    score += 1
        if score > 0:
            candidate_docs.append((doc, score))

    if candidate_docs:
        candidate_docs.sort(key=lambda x: x[1], reverse=True)
        selected_docs = [x[0] for x in candidate_docs[:prefilter_top]]
        print(f"[+] filtered docs: {len(selected_docs)}")
    else:
        print("[!] no tag match → using all docs")
        selected_docs = docs

    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )
    qvec = await safe_embed_call(embed_client, [question])
    qvec = np.array(qvec[0], dtype="float32").reshape(1, -1)

    vectors, doc_objs = [], []
    for d in selected_docs:
        if "embedding" not in d:
            continue
        vectors.append(d["embedding"])
        doc_objs.append(d)
    if not vectors:
        D, I = index.search(qvec, k)
        return [docs[i] for i in I[0]]

    vectors = np.array(vectors).astype("float32")
    sub_index = faiss.IndexFlatL2(vectors.shape[1])
    sub_index.add(vectors)

    D, I = sub_index.search(qvec, min(faiss_top, len(vectors)))
    faiss_docs = [doc_objs[i] for i in I[0]]
    print(f"[+] faiss top docs: {len(faiss_docs)}")

    print("[+] reranking...")
    res = rerank_docs(question, [d["text"] for d in faiss_docs], key=EMBEDDER_API_KEY)
    entries = res.get("results", []) if isinstance(res, dict) else []

    if entries:
        entries = sorted(entries, key=lambda x: x.get("relevance_score", 0), reverse=True)
        rerank_indices = [x["index"] for x in entries]
        final_docs = [faiss_docs[idx] for idx in rerank_indices[:k] if 0 <= idx < len(faiss_docs)]
        print(f"[+] final retrieved: {len(final_docs)}")
        return final_docs
    else:
        return faiss_docs[:k]

# ==============================
# ANSWERING
# ==============================
async def answer_generation(question):
    ctx_docs = await retrieve(question, k=3)
    ctx = "\n\n".join([c["text"] for c in ctx_docs])
    prompt = f"""
Ты — эксперт по финансовым вопросам. Отвечай строго на основе контекста ниже. 
Если в тексте нет прямого ответа, объясни максимально подробно, опираясь на доступную информацию, не придумывая фактов.
Если вопрос касается законов, сумм, сроков — используй только данные из контекста. Отвечай полно, развёрнуто и по делу.

Контекст:
{ctx}

Вопрос:
{question}

Ответ:
""".strip()

    llm_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )
    return await safe_llm_call(llm_client, prompt)

# ==============================
# MAIN с batch = 10
# ==============================
async def main():
    print("🔍 Проверяем наличие индекса...")
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        from build_index_module import build_index
        print("⚙️ Индекса нет → создаём...")
        await build_index()
    else:
        print("✅ Индекс найден")

    print("📁 Загружаем questions.csv ...")
    df = pd.read_csv("questions.csv")
    qs = df["Вопрос"].tolist()

    # загружаем кэш ответов
    if os.path.exists(ANSWERS_CACHE_PATH):
        with open(ANSWERS_CACHE_PATH, "r", encoding="utf-8") as f:
            answers_cache = json.load(f)
    else:
        answers_cache = {}

    llm_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )

    BATCH_SIZE = 20
    for i in range(0, len(qs), BATCH_SIZE):
        batch = qs[i:i+BATCH_SIZE]

        # пропускаем уже обработанные
        batch = [q for q in batch if q not in answers_cache]
        if not batch:
            continue

        # keywords для всего батча
        kw_tasks = [extract_query_keywords(q, llm_client) for q in batch]
        kws_list = await tqdm_asyncio.gather(*kw_tasks)
        for q, kws in zip(batch, kws_list):
            print(f"[+] keywords for question: {q} → {kws}")

        # ответы
        ans_tasks = [answer_generation(q) for q in batch]
        ans_list = await tqdm_asyncio.gather(*ans_tasks)

        for q, ans in zip(batch, ans_list):
            answers_cache[q] = ans

        # сохраняем прогресс
        with open(ANSWERS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(answers_cache, f, ensure_ascii=False, indent=2)

        # очистка памяти
        gc.collect()

    # финальный CSV
    df["Ответы на вопрос"] = df["Вопрос"].map(lambda q: answers_cache.get(q, ""))
    df.to_csv("submission.csv", index=False)
    print("✅ submission.csv готов")

if __name__ == "__main__":
    asyncio.run(main())



