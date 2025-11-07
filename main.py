import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
from rag.rag import retrieve

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")


def answer_generation(question):
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY,
    )

    context_list = retrieve(question, k=3)
    context = "\n\n".join(context_list)

    prompt = f"""
Ты — финансовый помощник.
Отвечай строго на вопрос, используя контекст.
Если данных недостаточно — скажи об этом корректно.

Вопрос:
{question}

Контекст:
{context}

Ответ:
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
    questions = pd.read_csv('./data/questions.csv')
    questions_list = questions['Вопрос'].tolist()

    answer_list = []
    for current_question in tqdm(questions_list, desc="Генерация ответов"):
        answer = answer_generation(question=current_question)
        answer_list.append(answer)

    questions['Ответы на вопрос'] = answer_list
    questions.to_csv('submission.csv', index=False)
