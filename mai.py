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
# MARKDOWN CLEANING (сохраняем заголовки ##)
# =====================================================
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


# =====================================================
# SAFE ASYNC REQUESTS
# =====================================================
async def safe_embed_call(client, texts, retries=5):
    """
    Асинхронный вызов эмбеддингов.
    Возвращает list[list[float]] или None при ошибке.
    """
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
                return None
            wait = 2 + attempt * 2 + random.random() * 2
            print(f"⚠️ embed retry {attempt}/{retries}, wait {wait:.1f}s")
            await asyncio.sleep(wait)


async def safe_llm_call(client, prompt, retries=5):
    """
    Асинхронный вызов LLM (чтобы получить текст ответа).
    """
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


# =====================================================
# PARSING / CHUNKING
# =====================================================
def parse_sections(doc_id, text, annotation, tags):
    """
    Разбиваем документ по '##' на секции с базовой метаинформацией.
    """
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
        if len(lines) == 1:
            title = lines[0].strip()
            body = ""
        else:
            title = lines[0].strip()
            body = lines[1].strip()
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


def chunk_section(section, chunk_size=800, overlap=80, min_len=100):
    """
    Разбиваем секцию на чанки с помощью RecursiveCharacterTextSplitter.
    Возвращаем список dict-чанков (без эмбеддингов).
    """
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


# =====================================================
# KEYWORD EXTRACTION (простая версия — regexp + stopwords + bigrams)
# =====================================================
RUS_STOPWORDS = {
    "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
    "его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было",
    "вот","от","меня","еще","нет","о","из","ему","теперь","когда","даже","ну","вдруг",
    "ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
    "уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где",
    "есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто",
    "чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому"
}


def extract_query_keywords(question: str, min_len=3):
    """
    Простая token-based обработка вопроса: токены + биграммы, без лемматизации.
    """
    if not isinstance(question, str) or not question.strip():
        return []
    text = question.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)
    lemmas = [t for t in tokens if len(t) >= min_len and t not in RUS_STOPWORDS]
    bigrams = [lemmas[i] + " " + lemmas[i + 1] for i in range(len(lemmas) - 1)]
    return list(set(lemmas + bigrams))


# =====================================================
# INDEX BUILDING
# =====================================================
async def build_index():
    """
    1) Загружает train_data.csv
    2) Разбивает на секции и чанки
    3) Генерирует эмбеддинги батчами
    4) Сохраняет FAISS index + docs.pkl (включая embedding в каждой записи)
    """
    print("📁 Загрузка train_data.csv ...")
    df = pd.read_csv("train_data.csv")

    print("🧠 Парсим документы...")
    all_chunks = []
    for row in df.itertuples():
        doc_id = getattr(row, "id")
        raw_tags = getattr(row, "tags")
        try:
            tags = json.loads(raw_tags.replace("'", '"'))
        except Exception:
            tags = []
        annotation = getattr(row, "annotation")
        text = clean_markdown_text(getattr(row, "text"))
        sections = parse_sections(doc_id, text, annotation, tags)
        for sec in sections:
            ch = chunk_section(sec)
            all_chunks.extend(ch)

    print(f"✅ total chunks: {len(all_chunks)}")

    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )

    print("⚡ async embedding...")
    vectors = []
    valid_chunks = []

    batch_size = 50
    tasks = []
    for i in range(0, len(all_chunks), batch_size):
        batch = [c["text"] for c in all_chunks[i:i + batch_size]]
        tasks.append(safe_embed_call(embed_client, batch))

    results = await tqdm_asyncio.gather(*tasks)

    # Собираем результаты, сохраняем embedding прямо в объект chunk
    idx = 0
    for r in results:
        batch = all_chunks[idx:idx + batch_size]
        if r is None:
            idx += batch_size
            continue
        for emb, chunk in zip(r, batch):
            chunk["embedding"] = emb
            vectors.append(emb)
            valid_chunks.append(chunk)
        idx += batch_size

    vectors = np.array(vectors).astype("float32")
    print(f"✅ valid vectors: {vectors.shape[0]}")

    if vectors.shape[0] == 0:
        raise RuntimeError("❌ Ни один документ не удалось векторизовать!")

    print("📦 building FAISS index...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(valid_chunks, f)

    print("✅ index + docs saved")


# =====================================================
# RERANK HELPER
# =====================================================
def rerank_docs(query, documents, key):
    """
    Позвоночник реранка: отправляем запрос на внешний /rerank и возвращаем parsed JSON.
    Важно: возвращаем dict с полем "results" (как у провайдера).
    """
    url = "https://ai-for-finance-hack.up.railway.app/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
        "query": query,
        "documents": documents
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print("❌ rerank request failed:", e)
        return {"results": []}

    try:
        data = resp.json()
        return data
    except Exception:
        print("❌ error parsing rerank response:", resp.text)
        return {"results": []}


# =====================================================
# RETRIEVAL (tag-prefilter + sub-FAISS + rerank)
# =====================================================
async def retrieve(question, k=3, prefilter_top=50, faiss_top=30):
    """
    1) извлекаем простые keywords
    2) фильтруем документы по тегам (soft prefilter)
    3) строим временный под-индекс из отобранных документов (с их embedding)
    4) делаем FAISS поиск + rerank (Qwen)
    5) возвращаем список dict-документов (top-k)
    """
    print("\n=== RETRIEVE ===")
    print(f"[+] question: {question}")

    kw = extract_query_keywords(question)
    print(f"[+] keywords: {kw}")

    # загружаем глобальный индекс и документы
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    # TAG SOFT-PREFILTER: считаем простое совпадение токенов тега и kw
    candidate_docs = []
    for doc in docs:
        tags = doc.get("tags", []) or []
        score = 0
        for t in tags:
            t_tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", str(t).lower())
            for tt in t_tokens:
                if tt in kw:
                    score += 2
                elif any(tt in term for term in kw):
                    score += 1
        if score > 0:
            candidate_docs.append((doc, score))

    if candidate_docs:
        candidate_docs.sort(key=lambda x: x[1], reverse=True)
        selected_docs = [x[0] for x in candidate_docs[:prefilter_top]]
        print(f"[+] tag-match docs: {len(selected_docs)}")
    else:
        print("[!] no tag match → using all docs")
        selected_docs = docs

    # embed query
    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )
    qvec = await safe_embed_call(embed_client, [question])
    if not qvec:
        print("❌ failed to embed question → returning empty")
        return []
    qvec = np.array(qvec[0], dtype="float32").reshape(1, -1)

    # build sub-index from selected_docs (use saved embeddings)
    print("[+] building temp index for filtered docs...")
    vectors = []
    doc_objs = []
    for d in selected_docs:
        if "embedding" in d and d["embedding"] is not None:
            vectors.append(d["embedding"])
            doc_objs.append(d)
        else:
            # логируем, но не падаем
            print("❌ doc without embedding (skipped):", d.get("doc_id", "unknown"))

    if not vectors:
        print("[!] fallback to global index (no embeddings in selected docs)")
        D, I = index.search(qvec, k)
        return [docs[i] for i in I[0]]

    vectors = np.array(vectors).astype("float32")
    sub_index = faiss.IndexFlatL2(vectors.shape[1])
    sub_index.add(vectors)

    D, I = sub_index.search(qvec, min(faiss_top, len(vectors)))
    faiss_docs = [doc_objs[i] for i in I[0]]
    print(f"[+] faiss top docs: {len(faiss_docs)}")

    # RERANK via external API
    print("[+] reranking...")
    texts_for_rerank = [d["text"] for d in faiss_docs]
    res = rerank_docs(query=question, documents=texts_for_rerank, key=EMBEDDER_API_KEY)

    entries = res.get("results", []) if isinstance(res, dict) else []
    if entries:
        # сортировка по relevance_score (если есть)
        if isinstance(entries[0], dict) and "relevance_score" in entries[0]:
            entries = sorted(entries, key=lambda x: x.get("relevance_score", 0), reverse=True)
        rerank_indices = [x["index"] for x in entries]
        print("[+] rerank indices:", rerank_indices[:k])

        final_docs = []
        for idx in rerank_indices[:k]:
            if 0 <= idx < len(faiss_docs):
                final_docs.append(faiss_docs[idx])
        print(f"[+] final retrieved: {len(final_docs)}")
        return final_docs
    else:
        print("[!] rerank empty → fallback faiss subset")
        return faiss_docs[:k]


# =====================================================
# ANSWERING (формирование prompt и вызов LLM)
# =====================================================
async def answer_generation(question):
    ctx_docs = await retrieve(question, k=3)
    # ctx_docs — список dict
    ctx = "\n\n".join([c["text"] for c in ctx_docs]) if ctx_docs else ""

    prompt = f"""
Ты — высококвалифицированный финансовый помощник.
Отвечай строго на основе контекста ниже.
Не придумывай факты, которых нет в тексте.
Если ответа в тексте нет — так и скажи: "К сожалению, я не могу ответить на данный вопрос."
Отвечай полно, развёрнуто и по делу. Язык: русский.

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


# =====================================================
# MAIN (входной/выходной контракт сохраняется)
# =====================================================
async def main():
    print("🔍 Проверяем наличие индекса...")
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        print("⚙️ Индекса нет → создаём...")
        await build_index()
    else:
        print("✅ Индекс найден")

    print("📁 Загружаем questions.csv ...")
    df = pd.read_csv("questions copy.csv")
    qs = df["Вопрос"].tolist()

    print("💬 Генерируем ответы...")
    tasks = [answer_generation(q) for q in qs]
    answers = await tqdm_asyncio.gather(*tasks)

    df["Ответы на вопрос"] = answers
    df.to_csv("submission.csv", index=False)
    print("✅ submission.csv готов")


if __name__ == "__main__":
    asyncio.run(main())
