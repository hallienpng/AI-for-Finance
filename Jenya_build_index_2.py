import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import re

def optimized_chunking_for_qa():
    print("Загрузка и анализ данных...")
    df = pd.read_csv('train_data.csv')
    
    chunks = []
    metadata = []
    
    for idx, row in df.iterrows():
        text = row['text']
        
        # Стратегия 1: Если текст содержит четкую структуру Q&A
        if is_structured_qa(text):
            qa_chunks = split_structured_qa(text, idx)
            chunks.extend(qa_chunks)
            metadata.extend([{'doc_id': idx, 'type': 'qa_pair'}] * len(qa_chunks))
        
        # Стратегия 2: Для обычных статей - умная нарезка
        else:
            article_chunks = split_article_text(text, idx)
            chunks.extend(article_chunks)
            metadata.extend([{'doc_id': idx, 'type': 'article'}] * len(article_chunks))
    
    print(f"Нарезано на {len(chunks)} чанков.")
    
    print("\nВекторизация...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    print(f"Созданы эмбеддинги размером: {embeddings.shape}")
    
    print("\nСоздание индекса FAISS...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    
    print("\nСохранение с метаданными...")
    faiss.write_index(index, 'train_data.index')
    
    with open('docs.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Сохраняем статистику
    save_chunk_statistics(chunks, metadata)
    
    print("База знаний успешно создана и сохранена!")

def is_structured_qa(text):
    """Определяем, имеет ли текст структуру вопрос-ответ"""
    qa_patterns = [
        r'вопрос:.+?ответ:', 
        r'q:.+?a:',
        r'вопрос\s*\d+[.:].+?ответ\s*\d+[.:]',
        r'•\s*вопрос:.+?•\s*ответ:'
    ]
    
    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.DOTALL) for pattern in qa_patterns)

def split_structured_qa(text, doc_id):
    """Специальная обработка для Q&A структур"""
    chunks = []
    
    # Паттерн для извлечения Q&A пар
    qa_pattern = r'(вопрос\s*\d*[.:]?\s*.+?)(?=ответ\s*\d*[.:]?\s*(.+?))(?=вопрос\s*\d|$)'
    
    matches = re.findall(qa_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        question_part = match[0].strip()
        answer_part = match[1].strip() if len(match) > 1 else ""
        
        # Создаем чанк с полной Q&A парой
        qa_chunk = f"Вопрос: {question_part}\n\nОтвет: {answer_part}"
        
        # Проверяем размер и при необходимости разбиваем ответ
        if len(qa_chunk) > 800:
            # Разбиваем только ответную часть
            answer_chunks = split_long_answer(answer_part)
            for answer_chunk in answer_chunks:
                chunks.append(f"Вопрос: {question_part}\n\nОтвет: {answer_chunk}")
        else:
            chunks.append(qa_chunk)
    
    return chunks

def split_article_text(text, doc_id):
    """Обработка обычных статей"""
    # Используем более подходящий сплиттер для русского текста
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        length_function=len,
        separators=['\n\n', '\n', '. ', '! ', '? ', ' ', '']
    )
    
    return text_splitter.split_text(text)

def split_long_answer(answer, max_length=400):
    """Разбивает длинные ответы на части"""
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def save_chunk_statistics(chunks, metadata):
    """Сохранение статистики по чанкам"""
    stats = {
        'total_chunks': len(chunks),
        'avg_chunk_length': np.mean([len(chunk) for chunk in chunks]),
        'chunk_types': pd.Series([m['type'] for m in metadata]).value_counts().to_dict(),
        'chunks_per_document': len(chunks) / len(set(m['doc_id'] for m in metadata))
    }
    
    print("\nСтатистика чанков:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    with open('chunk_stats.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(stats, f, ensure_ascii=False, indent=2)

# Дополнительная функция для тестирования качества чанков
def test_chunk_quality():
    """Тестирование качества созданных чанков"""
    with open('docs.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("\nТестирование качества чанков:")
    
    # Показываем примеры разных типов чанков
    qa_chunks = [chunk for chunk, meta in zip(chunks, metadata) if meta['type'] == 'qa_pair']
    article_chunks = [chunk for chunk, meta in zip(chunks, metadata) if meta['type'] == 'article']
    
    print(f"Q&A чанков: {len(qa_chunks)}")
    print(f"Статейных чанков: {len(article_chunks)}")
    
    # Показываем примеры
    if qa_chunks:
        print("\nПример Q&A чанка:")
        print(qa_chunks[0][:300] + "...")
    
    if article_chunks:
        print("\nПример статейного чанка:")
        print(article_chunks[0][:300] + "...")

# Запуск
if __name__ == "__main__":
    optimized_chunking_for_qa()
    test_chunk_quality()