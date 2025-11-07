import pandas as pd
from langchain_text_splitters import MarkdownTextSplitter # Умная нарезка по Markdown
from sentence_transformers import SentenceTransformer # Мощный Векторизатор
import faiss # База векторов
import numpy as np
import pickle # Для сохранения чанков
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os

print("Шаг 1: Загрузка данных...")
# Загружаем статьи
df = pd.read_csv('train_data.csv')
# Объединяем все статьи в один большой текст (если их много)
# Если у вас всего 350 строк, можно итерироваться, но для Markdown лучше так:
all_text = "\n\n".join(df['text'].dropna().tolist())

print("ШаG 2: 'Умная' нарезка (Chunking)...")
# Создаем 'нарезчик', который понимает Markdown (заголовки, списки)
# chunk_size=500 - это хороший баланс
# chunk_overlap=50 - чтобы не терять контекст на стыках
text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(all_text)

print(f"Нарезано на {len(chunks)} чанков.")

print("Шаг 3: Векторизация (Embedding)...")
# Загружаем мощную многоязычную модель. Она лучше базовых.
# Эта модель есть в requirements.txt (через sentence-transformers)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 
# Превращаем каждый чанк в вектор (набор цифр)
# Это может занять несколько минут.
embeddings = model.encode(chunks, show_progress_bar=True)

print(f"Созданы эмбеддинги размером: {embeddings.shape}")

print("Шаг 4: Создание индекса FAISS...")
# Получаем размерность вектора (например, 768)
d = embeddings.shape[1] 
# Создаем простую, но быструю базу FAISS
index = faiss.IndexFlatL2(d) 
# Добавляем наши векторы в базу
index.add(np.array(embeddings).astype('float32'))

print("Шаг 5: Сохранение...")
# 1. Сохраняем саму базу векторов
faiss.write_index(index, 'train_data.index')
# 2. Сохраняем текстовые чанки (чтобы по ID из FAISS доставать текст)
with open('docs.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print("База знаний (индекс) успешно создана и сохранена!")


# --- ЭТА СТРОКА БОЛЬШЕ НЕ НУЖНА ---
# from rag.rag import retrieve 

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")

# --- НОВЫЙ БЛОК: ЗАГРУЗКА RAG-КОМПОНЕНТОВ ---
# Мы загружаем "базу знаний" (созданную вашим 2-м скриптом) 
# один раз при запуске скрипта.

print("Загрузка RAG компонентов...")
try:
    # 1. Загружаем индекс FAISS
    faiss_index = faiss.read_index('train_data.index')
    
    # 2. Загружаем текстовые чанки (из .pkl)
    with open('docs.pkl', 'rb') as f:
        all_chunks = pickle.load(f)
        
    # 3. Загружаем ту же модель для векторизации (важно, чтобы она совпадала!)
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 
    
    print("RAG компоненты успешно загружены.")
except FileNotFoundError:
    print("ОШИБКА: Файлы 'train_data.index' или 'docs.pkl' не найдены.")
    print("Пожалуйста, сначала запустите 'build_index.py'!")
    exit()
# --- КОНЕЦ НОВОГО БЛОКА ---


# --- НОВАЯ ФУНКЦИЯ 'retrieve' ---
# Эта функция заменяет "черный ящик", который вы импортировали.
# Она выполняет поиск локально.

def retrieve(question: str, k=3):
    """
    Ищет k наиболее релевантных чанков в локальной базе FAISS.
    """
    
    # 1. Векторизуем ВОПРОС (той же моделью)
    question_embedding = embedding_model.encode([question])
    
    # 2. Ищем в FAISS
    # D - дистанции (насколько похожи), I - индексы (ID) чанков
    D, I = faiss_index.search(np.array(question_embedding).astype('float32'), k)
    
    # 3. Достаем ТЕКСТ чанков по их ID из нашего списка 'all_chunks'
    retrieved_chunks_text = [all_chunks[i] for i in I[0]]
    
    return retrieved_chunks_text
# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---


def answer_generation(question):
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )

    # Эта строка ТЕПЕРЬ вызывает вашу НОВУЮ локальную функцию retrieve
    context_list = retrieve(question, k=3)
    context = "\n\n".join(context_list)

    prompt = f"""Ты — высококвалифицированный финансовый помощник "Al for Finance Bank".
Твоя задача - дать четкий и точный ответ на вопрос клиента, используя *только* предоставленные ниже статьи из базы знаний.
Не придумывай ничего, чего нет в тексте.
Если в статьях нет ответа на вопрос, вежливо сообщи: "К сожалению, данной информации у меня нет, предлагаю связаться со специалистом на горячей линии."
Отвечай на русском языке в формате plaintext.

### База Знаний:
{context}

### Вопрос Клиента:
{question}

### Твой Ответ:
"""

    response = client.chat.completions.create(
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Убедитесь, что 'questions.csv' в правильной папке
    questions = pd.read_csv('questions.csv') 
    questions_list = questions['Вопрос'].tolist()

    answer_list = []
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        answer = answer_generation(question=current_question)
        answer_list.append(answer)

    questions['Ответы на вопрос'] = answer_list
    questions.to_csv('submission.csv', index=False)
    print("Работа завершена! Файл 'submission.csv' готов.")