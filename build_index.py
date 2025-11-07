import pandas as pd
from langchain_text_splitters import MarkdownTextSplitter # Умная нарезка по Markdown
from sentence_transformers import SentenceTransformer # Мощный Векторизатор
import faiss # База векторов
import numpy as np
import pickle # Для сохранения чанков

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