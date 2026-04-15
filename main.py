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
    raise ValueError("EMBEDDER_API_KEY not found")

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY not found")

EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "openrouter/x-ai/grok-3-mini"

INDEX_PATH = "train_data.index"
DOCS_PATH = "docs.pkl"


# MARKDOWN CLEANING
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


# SAFE ASYNC REQUESTS
async def safe_embed_call(client, texts, retries=5):
    for attempt in range(1, retries + 1):
        try:
            r = await client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [x.embedding for x in r.data]
        except Exception as e:
            if attempt == retries:
                print(f"EMBEDDING FAIL: {e}")
                return None
            wait = 2 + attempt * 2 + random.random() * 2
            print(f"embed retry {attempt}/{retries}, wait {wait:.1f}s")
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


# =====================================================
# SECTION PARSING
# =====================================================
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
# LLM-BASED KEYWORD EXTRACTION
# =====================================================
async def extract_query_keywords(question: str):
    prompt = f"""
Ты — ассистент по анализу текстов. 
По следующему вопросу выдели только ключевые слова и словосочетания, которые максимально отражают смысл вопроса. 
Приводи слова к основной форме (лемматизируй) — не учитывай склонения, окончания или разные типы слова. 
Например, "мошеннические списания" → "мошенничество". 
Не включай стоп-слова вроде "и", "в", "как", "на", "что". 
Каждое ключевое слово или словосочетание — отдельная фраза, не более 3 слов. 
Выводи результат в виде JSON-массива строк.

Вопрос:
{question}
"""
    llm_client = AsyncOpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )
    try:
        response = await safe_llm_call(llm_client, prompt)
        if not response:
            raise ValueError("пустой ответ LLM")

        # пробуем достать JSON из текста
        try:
            keywords = json.loads(response)
        except json.JSONDecodeError:
            # ищем первый JSON-массив в тексте
            match = re.search(r'\[.*\]', response, flags=re.DOTALL)
            if match:
                keywords = json.loads(match.group())
            else:
                raise

        if isinstance(keywords, list):
            return [k.lower() for k in keywords if isinstance(k, str)]
        else:
            return []

    except Exception as e:
        print("❌ Ошибка LLM при получении ключевых слов:", e)
        # fallback к простому regexp
        tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", question.lower())
        RUS_STOPWORDS = {
            "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так",
            "его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было",
            "вот","от","меня","еще","нет","о","из","ему","теперь","когда","даже","ну","вдруг",
            "ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять",
            "уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где",
            "есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто",
            "чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому"
        }
        return list(set([t for t in tokens if t not in RUS_STOPWORDS and len(t) > 2]))



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
# RETRIEVAL
# =====================================================
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


async def retrieve(question, k=3, prefilter_top=50, faiss_top=60):
    print("\n=== RETRIEVE ===")
    print(f"[+] question: {question}")

    kw = await extract_query_keywords(question)
    print(f"[+] keywords: {kw}")

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

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
        selected_docs = docs
        print("[!] no tag match → using all docs")

    embed_client = AsyncOpenAI(base_url="https://ai-for-finance-hack.up.railway.app/", api_key=EMBEDDER_API_KEY)
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


# =====================================================
# ANSWERING
# =====================================================
async def answer_generation(question):
    ctx_docs = await retrieve(question, k=3)
    ctx = "\n\n".join([c["text"] for c in ctx_docs])

    prompt = f"""
Ты — эксперт по финансовым инструментам.

Правила:

1) Основывай объяснения на предоставленных данных.
При отсутствии нужных сведений — отвечай общими принципами, аккуратно формулируя выводы.
2) Если вопрос включает ставки, суммы, сроки, нормативы — используй только известные данные; при их отсутствии — описывай общие практики без конкретных цифр.
3) Отвечай полно, понятно и по делу.
4) Формат ответа — markdown:
 -Заголовок ### с кратким названием темы
 -Структурированное объяснение: пункты, подпункты, абзацы
 - При необходимости — примеры
5) Короткий итог или вывод
6) Тон: профессиональный, спокойный, без воды и фантазий.

Контекст:
{ctx}

Вопрос:
{question}

Ответ:
""".strip()

    llm_client = AsyncOpenAI(base_url="https://ai-for-finance-hack.up.railway.app/", api_key=LLM_API_KEY)
    return await safe_llm_call(llm_client, prompt)


# =====================================================
# MAIN (с батчами и промежуточным сохранением)
# =====================================================
async def main():
    print("🔍 Проверяем наличие индекса...")
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        print("⚙️ Индекса нет → создаём...")
        await build_index()
    else:
        print("✅ Индекс найден")

    print("📁 Загружаем questions.csv ...")
    df = pd.read_csv("questions.csv")

    batch_size = 25
    all_answers = []

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]
        print(f"💬 Обрабатываем вопросы {start_idx+1}–{start_idx+len(batch)}...")

        tasks = [answer_generation(q) for q in batch["Вопрос"].tolist()]
        answers = await tqdm_asyncio.gather(*tasks)

        # Формируем записи с нужной структурой
        for idx, answer in zip(batch["ID вопроса"], answers):
            all_answers.append({
                "ID вопроса": idx,
                "Вопрос": batch.loc[batch["ID вопроса"]==idx, "Вопрос"].values[0],
                "Ответы на вопрос": answer
            })

        # Сохраняем промежуточно
        pd.DataFrame(all_answers).to_csv("submission.csv", index=False)
        print(f"✅ Батч {start_idx+1}–{start_idx+len(batch)} сохранён, память очищена.")

        # Очищаем временные данные для экономии RAM
        del tasks, answers, batch

    print("✅ Все вопросы обработаны, submission.csv готов")




if __name__ == "__main__":
    asyncio.run(main())

