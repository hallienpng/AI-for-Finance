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
        
        return response.data.embedding
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
        
        return response.choices.message.content
    except Exception as e:
        print(f"Error in chat completion: {e}")
        return None

def rerank_docs(query, documents, key):
    """
    Reranks documents using the specified API and logs the estimated cost.
    """
    url = "https://ai-for-finance-hack.up.railway.app/rerank" # Updated URL
    model = "deepinfra/Qwen/Qwen3-Reranker-4B"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    payload = {
        "query": query,
        "documents": documents
    }
    
    try:
        # --- Cost Estimation (Client-Side) ---
        # We must calculate tokens *before* sending, as the API might not return them.
        # We tokenize the query + all documents to get the total processed tokens.
        query_tokens = len(ENCODER.encode(query))
        doc_tokens = sum(len(ENCODER.encode(doc)) for doc in documents)
        total_input_tokens = query_tokens + doc_tokens
        
        cost = logger.calculate_cost(model, total_input_tokens, 0)
        
        # Log the *estimated* cost
        logger.log_cost(
            component="reranker",
            model_name=model,
            input_tokens=total_input_tokens,
            output_tokens=0,  # Rerankers don't have "output tokens" in the same way
            cost=cost
        )
        
        # --- Actual API Call ---
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error during reranking request: {e}")
        return None
    except Exception as e:
        print(f"An error occurred in rerank_docs: {e}")
        return None

# --- Example Usage ---

if __name__ == "__main__":
    
    # --- 0. Load Data ---
    print("--- 0. Loading data from CSVs ---")
    try:
        # Читаем ОБЫЧНЫЙ CSV, где разделитель - запятая
        df_train = pd.read_csv('train_data.csv', delimiter=',', skipinitialspace=True)
        # Убираем пробелы из имен колонок (на всякий случай)
        df_train.columns = df_train.columns.str.strip()

        # То же самое для файла с вопросами
        df_questions = pd.read_csv('questions.csv', delimiter=',', skipinitialspace=True)
        df_questions.columns = df_questions.columns.str.strip()

        # Теперь колонки 'text' и 'Вопрос' должны быть на месте
        docs_to_rank = df_train['text'].tolist()
        test_query = df_questions['Вопрос'].iloc[0]
        
        print(f"Data loaded. Found {len(docs_to_rank)} documents.")
        print(f"Test query: '{test_query}'\n")

    except FileNotFoundError as e:
        print(f"Error: CSV file not found. {e}")
        print("Please make sure 'train_data.csv' and 'questions.csv' are in the same directory.")
        exit()
    except KeyError as e:
        print(f"Error: Column not found. {e}")
        print("CSV-файлы были загружены, но колонка 'text' или 'Вопрос' не найдена.")
        print("Проверьте реальные имена колонок в файле. Вот что я вижу:")
        if 'df_train' in locals():
            print(f"Колонки train_data: {list(df_train.columns)}")
        if 'df_questions' in locals():
            print(f"Колонки questions: {list(df_questions.columns)}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # --- 1. Get an embedding ---
    print("--- 1. Testing Embedding API ---")
    my_text = "This is a test sentence for embedding."
    embedding = get_embedding(my_text)
    if embedding:
        print(f"Embedding successful (first 5 dims): {embedding[:5]}...\n")

    # --- 2. Rerank documents ---
    print("--- 2. Testing Reranker API ---")
    # Используем данные из CSV
    # ВАЖНО: Реранкер может иметь лимит на кол-во документов.
    # Для теста возьмем первые 10 документов.
    
    # !!! ИСПРАВЛЕН КЛЮЧ !!!
    # Reranker должен использовать свой ключ (DEEPINFRA), а не ключ эмбеддера
    reranked_results = rerank_docs(test_query, docs_to_rank[:10], key=EMBEDDER_API_KEY)
    
    if reranked_results:
        print("Reranking successful (top 3 results):")
        top_3 = reranked_results.get('results', [])[:3]
        print(json.dumps(top_3, indent=2, ensure_ascii=False))
        print("\n")

    # --- 3. Get a chat completion ---
    print("--- 3. Testing Chat Completion API ---")
    # Формируем messages на основе вопроса из CSV
    messages = [
        {"role": "system", "content": """Ты — высококвалифицированный финансовый помощник "Al for Finance Bank".
Твоя задача - дать четкий и точный ответ на вопрос клиента, используя *только* предоставленные ниже статьи из базы знаний.
Не придумывай ничего, чего нет в тексте.
Если в статьях нет ответа на вопрос, вежливо сообщи: "К сожалению, данной информации у меня нет, предлагаю связаться со специалистом на горячей линии."
Отвечай на русском языке парграфом текста без форматирований."""},
        {"role": "user", "content": test_query}
    ]
    chat_response = get_chat_completion(messages, model=LLM_MODELS['mistral'])
    if chat_response:
        print(f"Chatbot response: {chat_response}\n")

    # --- 4. Check the log file ---
    print(f"--- 4. Review 'money_used.txt' for cost breakdown ---")
    try:
        with open("money_used.txt", "r", encoding="utf-8") as f:
            print(f.read())
    except FileNotFoundError:
        print("Log file not created yet (perhaps all API calls failed).")