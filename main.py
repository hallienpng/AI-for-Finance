import json
import requests
import tiktoken
import datetime
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# --- 1. Configuration: Pricing and API Keys ---

# Pricing per 1 Million tokens
PRICING_CATALOG = {
    "text-embedding-3-small": {
        "input": 0.02,
        "output": 0.0,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "deepinfra/Qwen/Qwen3-Reranker-4B": {
        "input": 0.025,  # Assuming reranker cost is based on total tokens processed
        "output": 0.0,
    },

    "openrouter/mistralai/mistral-small-3.2-24b-instruct": {
        "input": 0.06, # Уточните актуальную цену
        "output": 0.18 # Уточните актуальную цену
    },
    "openrouter/meta-llama/llama-3-70b-instruct": {
        "input": 0.3, # Уточните актуальную цену
        "output": 0.4 # Уточните актуальную цену
    },
    "openrouter/x-ai/grok-3-mini": {
        "input": 0.3, # Уточните актуальную цену
        "output": 0.5 # Уточните актуальную цену
    },
    "openrouter/google/gemma-3-27b-it": {
        "input": 0.09, # Уточните актуальную цену
        "output": 0.16 # Уточните актуальную цену
    },

}

# Setup tiktoken encoder for cost calculation
# cl100k_base is used by text-embedding-3 and gpt-4o-mini
ENCODER = tiktoken.get_encoding("cl100k_base")

# --- 2. Cost Logging Class ---

class CostLogger:
    def __init__(self, log_file="money_used.txt"):
        self.log_file = log_file

    def calculate_cost(self, model_name, input_tokens, output_tokens=0):
        """Calculates the cost based on the model and token counts."""
        if model_name not in PRICING_CATALOG:
            print(f"Warning: Price for model '{model_name}' not found. Cost will be 0.")
            return 0.0

        prices = PRICING_CATALOG[model_name]
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        return input_cost + output_cost

    def log_cost(self, component, model_name, input_tokens, output_tokens, cost):
        """Appends a new cost entry to the log file."""
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "component": component,
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n") # Добавлено ensure_ascii=False
        except IOError as e:
            print(f"Error writing to log file: {e}")

# Initialize the logger
logger = CostLogger()

# --- 3. Wrapped API Functions with Cost Tracking ---

def get_embedding(text_to_embed, model=EMBEDDER_MODELS['small']):
    """
    Gets embedding for text and logs the cost.
    Uses the official OpenAI client.
    """
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=EMBEDDER_API_KEY,
    )
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text_to_embed
        )
        
        # Extract usage from the response
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        
        # Calculate cost
        cost = logger.calculate_cost(model, prompt_tokens, 0)
        
        # Log the transaction
        logger.log_cost(
            component="embedding",
            model_name=model,
            input_tokens=prompt_tokens,
            output_tokens=0,
            cost=cost
        )
        
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_chat_completion(messages, model=LLM_MODELS['mistral']):
    """
    Gets a chat completion from OpenAI and logs the cost.
    """
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=LLM_API_KEY
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        # Extract usage from the response
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Calculate cost
        cost = logger.calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Log the transaction
        logger.log_cost(
            component="llm_chat",
            model_name=model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost=cost
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat completion: {e}")
        return None

def rerank_docs(query, documents, key):
    """
    Reranks documents using the specified API and logs the estimated cost.
    """
    url = "https://ai-for-finance-hack.up.railway.app/rerank"
    
    model = "deepinfra/Qwen/Qwen3-Reranker-4B"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    # 2. ТЕПЕРЬ 'model' можно безопасно добавить в payload
    payload = {
        "model": model,  # <-- Вот исправление, которое я предлагал
        "query": query,
        "documents": documents
    }
    
    try:
        # --- Cost Estimation (Client-Side) ---
        query_tokens = len(ENCODER.encode(query))
        doc_tokens = sum(len(ENCODER.encode(doc)) for doc in documents)
        total_input_tokens = query_tokens + doc_tokens
        
        cost = logger.calculate_cost(model, total_input_tokens, 0)
        
        # Log the *estimated* cost
        logger.log_cost(
            component="reranker",
            model_name=model,
            input_tokens=total_input_tokens,
            output_tokens=0,
            cost=cost
        )
        
        # --- Actual API Call ---
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error during reranking request: {e}")
        return None
    except Exception as e:
        print(f"An error occurred in rerank_docs: {e}")
        return None

# --- Example Usage ---

# --- Example Usage (Production RAG Pipeline) ---

if __name__ == "__main__":
    
    # --- 0. Load Data ---
    print("--- 0. Loading data from CSVs ---")
    try:
        # Читаем ОБЫЧНЫЙ CSV, где разделитель - запятая
        df_train = pd.read_csv('train_data.csv', delimiter=',', skipinitialspace=True)
        df_train.columns = df_train.columns.str.strip()
        # Это все наши документы (контекст)
        all_documents = df_train['text'].tolist()

        # То же самое для файла с вопросами
        df_questions = pd.read_csv('questions.csv', delimiter=',', skipinitialspace=True)
        df_questions.columns = df_questions.columns.str.strip()

        print(f"Data loaded. Found {len(all_documents)} documents for context.")
        print(f"Found {len(df_questions)} questions to process for submission.csv\n")

    except FileNotFoundError as e:
        print(f"Error: CSV file not found. {e}")
        print("Please make sure 'train_data.csv' and 'questions.csv' are in the same directory.")
        exit()
    except KeyError as e:
        print(f"Error: Column not found. {e}")
        print("CSV-файлы были загружены, но колонка 'text' или 'Вопрос'/'ID вопроса' не найдена.")
        if 'df_train' in locals():
            print(f"Колонки train_data: {list(df_train.columns)}")
        if 'df_questions' in locals():
            print(f"Колонки questions: {list(df_questions.columns)}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # Этот промпт будет использоваться для КАЖДОГО вопроса
    SYSTEM_PROMPT = """Ты — высококвалифицированный финансовый помощник "Al for Finance Bank".
Твоя задача - дать четкий и точный ответ на вопрос клиента, используя *только* предоставленные ниже статьи из базы знаний.
Не придумывай ничего, чего нет в тексте.
Если в статьях нет ответа на вопрос, вежливо сообщи: "К сожалению, данной информации у меня нет, предлагаю связаться со специалистом на горячей линии."
Отвечай на русском языке парграфом текста без форматирований."""

    # --- 2. Process all questions and create submission ---
    print("--- 2. Starting RAG pipeline to generate submission.csv ---")
    
    # Здесь будем хранить результаты
    results_list = []
    
    # Идем по каждому вопросу в df_questions
    for index, row in df_questions.iterrows():
        question_id = row['ID вопроса']
        query = row['Вопрос']
        
        print(f"\nProcessing question ID: {question_id} ({index + 1}/{len(df_questions)})...")
        print(f"  Query: {query[:70]}...")

        # --- RAG Step 1: Retrieve & Rerank ---
        # Ранжируем ВСЕ документы, чтобы найти лучшие
        print("  Reranking...")
        reranked_results = rerank_docs(query, all_documents, key=EMBEDDER_API_KEY)

        print("DEBUG: Reranker response structure (first result):")
        if reranked_results and 'results' in reranked_results and len(reranked_results['results']) > 0:
            print(json.dumps(reranked_results['results'][0], indent=2, ensure_ascii=False))
        else:
            print("DEBUG: Reranker response is empty or invalid.")
        
        chat_response = "" # Ответ по умолчанию
        
        if reranked_results and 'results' in reranked_results:
            # --- RAG Step 2: Augment ---
            # Берем ТОП-3 лучших документа
            top_docs = reranked_results.get('results', [])[:3]
            
            # 1. Сначала получаем ИНДЕКСЫ из ответа реранкера
            top_docs_indices = [res['index'] for res in top_docs]
            # 2. По индексам достаем ТЕКСТ из нашего полного списка
            top_docs_text = [all_documents[i] for i in top_docs_indices]
            
            # Соединяем их в один большой текст (контекст)
            context = "\n\n---\n\n".join(top_docs_text)
            
            print(f"  Found {len(top_docs)} relevant documents.")

            # --- RAG Step 3: Generate ---
            # Собираем промпт для LLM, ВСТАВЛЯЯ КОНТЕКСТ
            user_content = f"""Вот статьи из базы знаний:
            
{context}

---

Вопрос клиента: {query}"""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            print("  Generating answer...")
            chat_response = get_chat_completion(messages, model=LLM_MODELS['mistral'])
            
            if chat_response is None:
                print("  Error: Chat completion failed.")
                chat_response = "ОШИБКА: Не удалось сгенерировать ответ."
            else:
                print(f"  Response: {chat_response[:70]}...")
        
        else:
            print("  Error: Reranking failed or returned no results.")
            chat_response = "ОШИБКА: Не удалось найти релевантные документы."

        # --- RAG Step 4: Store ---
        # Добавляем результат в наш список
        results_list.append({
            "ID вопроса": question_id,
            "Вопрос": query,
            "Ответы на вопрос": chat_response
        })

    # --- 3. Save to CSV ---
    print("\n--- 3. Saving submission file ---")
    try:
        df_submission = pd.DataFrame(results_list)
        # index=False, чтобы не добавлять лишнюю колонку с индексами
        df_submission.to_csv("submission.csv", index=False)
        print("Successfully saved results to submission.csv")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # --- 4. Check the log file ---
    print(f"\n--- 4. Review 'money_used.txt' for cost breakdown ---")
    try:
        with open("money_used.txt", "r", encoding="utf-8") as f:
            print(f.read())
    except FileNotFoundError:
        print("Log file not created yet.")