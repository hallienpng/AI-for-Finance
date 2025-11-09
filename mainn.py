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
# MARKDOWN CLEANING
# (НЕ удаляем заголовки ## — они важны!)
# =====================================================
def clean_markdown_text(text: str):
    if not isinstance(text, str):
        return text

    # удаляем жир/курсив
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # удаляем картинки
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # ссылки → оставить только текст
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # инлайновый код
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)

    # много переводов строк
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# =====================================================
# SAFE ASYNC REQUESTS
# =====================================================
async def safe_embed_call(client, texts, retries=5):
    """
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
    for attempt in range(1, retries + 1):
        try:
            r = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
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
# SECTION PARSING
# =====================================================
def parse_sections(doc_id, text, annotation, tags):
    """
    Разбиваем текст на секции по "## ".
    """
    sections = []
    parts = re.split(r'\n##\s*', text)

    # Прелюдия до первого ## — если есть
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
        # первая строка — заголовок, далее текст
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


# =====================================================
# CHUNKING
# =====================================================
def chunk_section(section, chunk_size=800, overlap=80, min_len=100):
    """
    Разбиваем каждую секцию на маленькие чанки.
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

# правим выделение нужных слов
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
    text = question.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)

    # фильтр стоп-слов
    lemmas = [
        t for t in tokens
        if len(t) >= min_len and t not in RUS_STOPWORDS
    ]

    # биграммы
    bigrams = []
    for i in range(len(lemmas) - 1):
        bigrams.append(lemmas[i] + " " + lemmas[i + 1])

    return list(set(lemmas + bigrams))


# =====================================================
# INDEX BUILDING
# =====================================================
async def build_index():
    print("📁 Загрузка train_data.csv ...")
    df = pd.read_csv("train_data.csv")

    print("🧠 Парсим документы...")
    all_chunks = []

    for row in df.itertuples():
        doc_id = getattr(row, "id")

        # tags — строка → пытаемся распарсить
        raw_tags = getattr(row, "tags")
        try:
            tags = json.loads(raw_tags.replace("'", '"'))
        except:
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

    # Собираем только удачные
    idx = 0
    for r in results:
        batch = all_chunks[idx:idx + batch_size]
        if r is None:
            # пропускаем batch
            idx += batch_size
            continue

        for emb, chunk in zip(r, batch):
            chunk["embedding"] = emb              # ✅ сохраняем embedding внутрь doc
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
# RETRIEVAL (soft tag + embedding scoring)
# =====================================================
def normalize_tokens(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9]+", " ", text, flags=re.IGNORECASE)
    return text.split()


def count_tag_matches(question_tokens, tags):
    if not tags:
        return 0
    score = 0
    tags_norm = []
    for t in tags:
        t = t.lower()
        t = re.sub(r"[^a-zа-яё0-9]+", " ", t)
        tags_norm.extend(t.split())

    # пересечение ключевых слов
    for tok in question_tokens:
        if tok in tags_norm:
            score += 1

    return score

# РЕРАНКЕР
def rerank_docs(query, documents, key):
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

    resp = requests.post(url, headers=headers, json=payload)

    try:
        data = resp.json()
        return data
    except Exception:
        print("❌ error in rerank_docs response:", resp.text)
        return { "results": [] }



async def retrieve(question, k=3, prefilter_top=50, faiss_top=30):
    """
    1) Извлекаем keyword
    2) Фильтруем документы по тегам
    3) Берём top-N по FAISS
    4) Делаем rerank
    """

    print("\n=== RETRIEVE ===")
    print(f"[+] question: {question}")

    # ===== KEYWORDS =====
    kw = extract_query_keywords(question)
    print(f"[+] keywords: {kw}")

    # ===== LOAD INDEX =====
    index = faiss.read_index(INDEX_PATH)

    # ===== LOAD DOCS =====
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
                # частичное
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

    # ===== EMBED QUERY =====
    embed_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY
    )

    qvec = await safe_embed_call(embed_client, [question])
    qvec = np.array(qvec[0], dtype="float32").reshape(1, -1)

    # ===== BUILD SUB-INDEX =====
    print("[+] building temp index for filtered docs...")

    vectors = []
    doc_objs = []

    for d in selected_docs:
        if "embedding" not in d:
            print("❌ doc without embedding! Please embed before.")
            continue
        vectors.append(d["embedding"])
        doc_objs.append(d)

    if not vectors:
        print("[!] fallback to global index")
        D, I = index.search(qvec, k)
        return [docs[i] for i in I[0]]   # ← возвращаем dict, не text

    vectors = np.array(vectors).astype("float32")
    sub_index = faiss.IndexFlatL2(vectors.shape[1])
    sub_index.add(vectors)

    # ===== FAISS SEARCH =====
    D, I = sub_index.search(qvec, min(faiss_top, len(vectors)))
    faiss_docs = [doc_objs[i] for i in I[0]]

    print(f"[+] faiss top docs: {len(faiss_docs)}")

    # ===== RERANK (Qwen) =====
    print("[+] reranking...")

    res = rerank_docs(
        query=question,
        documents=[d["text"] for d in faiss_docs],
        key=EMBEDDER_API_KEY
    )

    entries = res.get("results", []) if isinstance(res, dict) else []

    if entries:
        # сортировка по релевантности
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
# ANSWERING
# =====================================================
async def answer_generation(question):
    ctx_docs = await retrieve(question, k=3)
    ctx = "\n\n".join([c["text"] for c in ctx_docs])

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
# MAIN
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
