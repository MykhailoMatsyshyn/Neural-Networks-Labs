"""
BACKEND ДЛЯ МУЛЬТИМОДАЛЬНОГО AI СЕРВІСУ
Поєднує: RAG + Agent + Image Generation + Image Analysis
"""

import os
import json
import base64
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
from email.mime.text import MIMEText
import pandas as pd

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google API бібліотеки не встановлено. Google тули будуть недоступні.")
    print("   Встановіть: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

# Завантажити змінні оточення
load_dotenv()

# Спробувати імпортувати chromadb (опціонально)
try:
    import chromadb  # type: ignore
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB не встановлено. RAG функціональність буде недоступна.")
    print("   Встановіть: pip install chromadb")
    print("   Або встановіть Visual C++ Build Tools для Windows")

# ==================== CONFIGURATION ====================
# Визначити, який API використовувати
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "false").lower() == "true"

# Якщо API ключ не знайдено в змінних оточення, можна встановити тут
# (для швидкого тестування - не рекомендується для production)
if not OPENAI_API_KEY:
    # Розкоментуйте наступний рядок та вставте ваш API ключ:
    os.environ["OPENAI_API_KEY"] = "..."
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ==================== INIT ====================
app = FastAPI()

# Middleware для вимкнення буферизації для streaming
@app.middleware("http")
async def no_cache_middleware(request, call_next):
    response = await call_next(request)
    if "text/event-stream" in response.headers.get("content-type", ""):
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ініціалізувати OpenAI клієнт
client = None
if USE_LM_STUDIO:
    # Використовувати LM Studio (локальна модель)
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        print(f"✅ Підключено до LM Studio: {LM_STUDIO_URL}")
    except Exception as e:
        print(f"⚠️  Помилка підключення до LM Studio: {e}")
        print("   Переконайтеся, що LM Studio запущено")
elif OPENAI_API_KEY:
    # Використовувати реальний OpenAI API
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ Підключено до OpenAI API")
        print(f"   Використовується API ключ: {OPENAI_API_KEY[:20]}...")
    except Exception as e:
        print(f"⚠️  Помилка ініціалізації OpenAI клієнта: {e}")
else:
    print("⚠️  OpenAI API ключ не знайдено!")
    print("   Варіанти:")
    print("   1. Створіть .env файл з OPENAI_API_KEY=your-key")
    print("   2. Встановіть змінну оточення: export OPENAI_API_KEY=your-key")
    print("   3. Встановіть USE_LM_STUDIO=true для використання LM Studio")
    print("   4. Розкоментуйте рядок в main.py для швидкого тестування")

# Ініціалізувати ChromaDB тільки якщо доступний
if CHROMADB_AVAILABLE:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("documents")
else:
    chroma_client = None
    collection = None

# ==================== CONVERSATION HISTORY STORAGE ====================
# Зберігати історію розмов для кожного thread
conversation_history = {}  # {thread_id: [messages]}

# ==================== ASSISTANTS API STORAGE ====================
# Зберігати OpenAI Assistant IDs та Thread IDs
assistants_cache = {}  # {thread_id: {"assistant_id": str, "openai_thread_id": str}}
vector_stores = {}  # {thread_id: vector_store_id}

# ==================== MODELS ====================
class ChatRequest(BaseModel):
    thread_id: str
    message: str
    mode: str = "chat"  # chat, image-gen, image-analyze
    image_base64: Optional[str] = None
    settings: dict = {}
    history: Optional[List[dict]] = []  # Історія попередніх повідомлень


class ChatResponse(BaseModel):
    content: str
    tools: list = []
    image_url: Optional[str] = None


# ==================== RAG FUNCTIONS (OpenAI File Search API) ====================
async def upload_file_to_openai(file_content: bytes, filename: str) -> Optional[str]:
    """Завантажити файл в OpenAI для File Search"""
    if not client or USE_LM_STUDIO:
        return None
    
    try:
        # Створити тимчасовий файл
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Завантажити в OpenAI
        with open(tmp_path, "rb") as f:
            file = client.files.create(
                file=f,
                purpose="assistants"
            )
        
        os.unlink(tmp_path)
        return file.id
    except Exception as e:
        print(f"⚠️  Помилка завантаження файлу в OpenAI: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None


async def create_or_get_vector_store(thread_id: str) -> Optional[str]:
    """Створити або отримати Vector Store для thread"""
    if not client or USE_LM_STUDIO:
        return None
    
    if thread_id in vector_stores:
        return vector_stores[thread_id]
    
    try:
        # Перевірити чи підтримується vector_stores API
        if not hasattr(client.beta, 'vector_stores'):
            print(f"⚠️  Vector Stores API не доступний в цій версії OpenAI SDK")
            return None
        
        vector_store = client.beta.vector_stores.create(
            name=f"Documents_{thread_id}",
        )
        vector_stores[thread_id] = vector_store.id
        return vector_store.id
    except AttributeError as e:
        print(f"⚠️  Vector Stores API не підтримується: {e}")
        print(f"   Використовується fallback до ChromaDB")
        return None
    except Exception as e:
        print(f"⚠️  Помилка створення Vector Store: {e}")
        return None


async def add_file_to_vector_store(file_id: str, vector_store_id: str):
    """Додати файл до Vector Store"""
    if not client or USE_LM_STUDIO:
        return False
    
    try:
        client.beta.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_id
        )
        return True
    except Exception as e:
        print(f"⚠️  Помилка додавання файлу до Vector Store: {e}")
        return False


# ==================== RAG FUNCTIONS (ChromaDB) ====================
# ChromaDB - це векторна база даних для зберігання документів та пошуку за схожістю.
# Вона конвертує текст у вектори (embeddings) та дозволяє швидко знаходити релевантні документи.
# Документи розбиваються на chunks (шматки) для кращого пошуку та індексації.
def add_documents_to_rag(docs: list):
    """Додати документи у векторну базу (ChromaDB) з chunking"""
    if not CHROMADB_AVAILABLE or not collection:
        return {"error": "ChromaDB не доступний. Встановіть: pip install chromadb"}
    
    try:
        # Chunking: розбити великі документи на менші частини
        CHUNK_SIZE = 1000  # Символів на chunk
        CHUNK_OVERLAP = 200  # Перекриття між chunks
        
        all_chunks = []
        for doc in docs:
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            
            if not text or not text.strip():
                continue
            
            # Якщо документ короткий, додати як є
            if len(text) <= CHUNK_SIZE:
                all_chunks.append({
                    "text": text,
                    "source": source,
                    "chunk_index": 0
                })
            else:
                # Розбити на chunks з перекриттям
                start = 0
                chunk_index = 0
                while start < len(text):
                    end = start + CHUNK_SIZE
                    chunk_text = text[start:end]
                    
                    # Знайти найближчий пробіл для кращого розбиття
                    if end < len(text):
                        last_space = chunk_text.rfind(' ')
                        if last_space > CHUNK_SIZE * 0.5:  # Якщо пробіл не дуже далеко
                            chunk_text = chunk_text[:last_space]
                            end = start + last_space
                    
                    all_chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "chunk_index": chunk_index
                    })
                    
                    # Переміститися з перекриттям
                    start = end - CHUNK_OVERLAP
                    chunk_index += 1
        
        # Додати всі chunks до ChromaDB
        if all_chunks:
            documents = [chunk["text"] for chunk in all_chunks]
            metadatas = [{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} for chunk in all_chunks]
            ids = [f"{chunk['source']}_chunk_{chunk['chunk_index']}_{datetime.now().timestamp()}" for chunk in all_chunks]
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            
            print(f"✅ Додано {len(all_chunks)} chunks з {len(docs)} документів до ChromaDB")
            return {"success": True, "chunks": len(all_chunks), "documents": len(docs)}
        else:
            print("⚠️  Немає тексту для додавання до ChromaDB")
            return {"error": "Немає тексту для додавання"}
            
    except Exception as e:
        print(f"❌ Помилка додавання документів до ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def retrieve_relevant_docs(query: str, n_results: int = 3):
    """Витягнути релевантні документи з метаданими (ChromaDB)"""
    if not CHROMADB_AVAILABLE or not collection:
        return []
    
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []
        
        # Повернути список словників з текстом та метаданими
        docs_with_metadata = []
        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            doc_id = ids[i] if i < len(ids) else f"doc_{i}"
            docs_with_metadata.append({
                "text": doc_text,
                "source": metadata.get("source", "unknown"),
                "id": doc_id
            })
        
        return docs_with_metadata
    except Exception as e:
        print(f"⚠️  Помилка пошуку в ChromaDB: {e}")
        return []


# ==================== GOOGLE API CONFIGURATION ====================
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "./token.json")
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.send'
]

# Глобальний каталог товарів
PRODUCT_CATALOG = {"iphone 15": 900, "macbook pro": 2500}

def get_google_service(service_name, version):
    """Отримати Google API сервіс"""
    if not GOOGLE_AVAILABLE:
        return None
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
            return build(service_name, version, credentials=creds)
        except Exception as e:
            print(f"⚠️  Помилка завантаження токену: {e}")
            return None
    return None

# ==================== AGENT TOOLS ====================
def get_item_price(item_name: str):
    """Отримати ціну товару з каталогу"""
    global PRODUCT_CATALOG
    key = item_name.lower().strip()
    price = PRODUCT_CATALOG.get(key)
    if price:
        return json.dumps({"item": item_name, "price": price})
    # Спробувати знайти частковий збіг
    for k, v in PRODUCT_CATALOG.items():
        if k in key or key in k:
            return json.dumps({"item": k, "price": v})
    return json.dumps({"error": "Not found"})


def calculate_shipping(destination: str, price: int):
    """Розрахувати вартість доставки"""
    base_shipping = 50
    percentage = int(price) * 0.05
    total = base_shipping + percentage
    return json.dumps({"destination": destination, "base_price": price, "shipping": total, "total": int(price) + total})


def book_meeting(topic: str, datetime_str: str, participants: str = ""):
    """Забронювати зустріч через Google Calendar"""
    if not GOOGLE_AVAILABLE:
        return json.dumps({"error": "Google API не доступний"})
    
    service = get_google_service('calendar', 'v3')
    if not service:
        return json.dumps({"error": "Auth failed. Переконайтеся, що token.json існує"})
    
    try:
        start_dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00') if 'Z' in datetime_str else datetime_str)
        if start_dt.tzinfo is None:
            # Якщо немає timezone, додаємо локальний
            from datetime import timezone
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        
        event = {
            'summary': topic,
            'description': f"Participants: {participants}" if participants else topic,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'Europe/Kiev',
            },
            'end': {
                'dateTime': (start_dt + timedelta(hours=1)).isoformat(),
                'timeZone': 'Europe/Kiev',
            },
        }
        res = service.events().insert(calendarId='primary', body=event).execute()
        return json.dumps({
            "status": "success",
            "link": res.get('htmlLink'),
            "event_id": res.get('id'),
            "topic": topic
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def send_email(recipient: str, subject: str, body: str):
    """Відправити email через Gmail"""
    if not GOOGLE_AVAILABLE:
        return json.dumps({
            "error": "Google API не доступний. Встановіть: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
        })
    
    service = get_google_service('gmail', 'v1')
    if not service:
        error_msg = "Google авторизація не налаштована. "
        error_msg += "Потрібно: 1) Завантажити credentials.json, 2) Запустити python auth_google.py, 3) Отримати token.json"
        return json.dumps({"error": error_msg})
    
    try:
        msg = MIMEText(body)
        msg['to'] = recipient
        msg['subject'] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = service.users().messages().send(userId='me', body={'raw': raw}).execute()
        return json.dumps({
            "status": "sent",
            "id": sent['id'],
            "to": recipient,
            "subject": subject,
            "message": f"Email успішно відправлено на {recipient}"
        })
    except Exception as e:
        return json.dumps({"error": f"Помилка відправки email: {str(e)}"})


available_functions = {
    "get_item_price": get_item_price,
    "calculate_shipping": calculate_shipping,
    "book_meeting": book_meeting,
    "send_email": send_email,
}

# Базовий список всіх доступних тулів
all_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_item_price",
            "description": "Get product price from catalog. Use this when user asks about product prices.",
            "parameters": {
                "type": "object",
                "properties": {"item_name": {"type": "string"}},
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": "Calculate shipping cost for a product. Requires destination and price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "price": {"type": "integer"},
                },
                "required": ["destination", "price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Schedule a meeting in Google Calendar. Requires topic and datetime in ISO format (YYYY-MM-DDTHH:MM:SS).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "datetime_str": {"type": "string"},
                    "participants": {"type": "string"},
                },
                "required": ["topic", "datetime_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email through Gmail. Requires recipient, subject, and body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["recipient", "subject", "body"],
            },
        },
    },
]

def get_enabled_tools(settings: dict):
    """Отримати список увімкнених тулів на основі налаштувань"""
    enabled_tools = []
    tool_settings = settings.get("enabledTools", {})
    
    # За замовчуванням всі тули увімкнені
    if tool_settings.get("get_item_price", True):
        enabled_tools.append(all_tools_schema[0])
    if tool_settings.get("calculate_shipping", True):
        enabled_tools.append(all_tools_schema[1])
    if tool_settings.get("book_meeting", True):
        enabled_tools.append(all_tools_schema[2])
    if tool_settings.get("send_email", True):
        enabled_tools.append(all_tools_schema[3])
    
    return enabled_tools if enabled_tools else all_tools_schema


# ==================== MODEL SELECTION LOGIC ====================
def normalize_messages(messages: List[dict], system_prompt: str) -> List[dict]:
    """Нормалізувати messages для OpenAI API"""
    if not messages:
        return [{"role": "system", "content": system_prompt}]
    
    normalized = []
    
    # Перше повідомлення - system prompt, має бути рядком
    MAX_SYSTEM_PROMPT_LENGTH = 10000  # Максимальна довжина system prompt
    if messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, list):
            # Якщо масив, витягти текст
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            system_content = " ".join(text_parts) if text_parts else system_prompt
            # Обрізати якщо занадто довге
            if len(system_content) > MAX_SYSTEM_PROMPT_LENGTH:
                system_content = system_content[:MAX_SYSTEM_PROMPT_LENGTH] + "... [system prompt обрізано]"
            normalized.append({
                "role": "system",
                "content": system_content
            })
        elif isinstance(content, str):
            # Обрізати якщо занадто довге
            if len(content) > MAX_SYSTEM_PROMPT_LENGTH:
                content = content[:MAX_SYSTEM_PROMPT_LENGTH] + "... [system prompt обрізано]"
            normalized.append({"role": "system", "content": content})
        else:
            normalized.append({"role": "system", "content": system_prompt})
    else:
        normalized.append({"role": "system", "content": system_prompt})
    
    # Інші повідомлення
    for msg in messages[1:]:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "user":
            # User message може мати content як рядок або масив (для images)
            if isinstance(content, list):
                # Перевірити чи це правильний формат для multimodal
                if len(content) > 0 and isinstance(content[0], dict) and "type" in content[0]:
                    # Правильний формат для multimodal, залишити як є
                    normalized.append({"role": "user", "content": content})
                else:
                    # Неправильний формат, конвертувати в рядок
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    text = " ".join(text_parts)
                    # Обрізати якщо занадто довге
                    MAX_MESSAGE_LENGTH = 8000
                    if len(text) > MAX_MESSAGE_LENGTH:
                        text = text[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                    normalized.append({"role": "user", "content": text})
            elif isinstance(content, str):
                # Обрізати якщо занадто довге
                MAX_MESSAGE_LENGTH = 8000
                if len(content) > MAX_MESSAGE_LENGTH:
                    content = content[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                normalized.append({"role": "user", "content": content})
            else:
                content_str = str(content) if content else ""
                MAX_MESSAGE_LENGTH = 8000
                if len(content_str) > MAX_MESSAGE_LENGTH:
                    content_str = content_str[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                normalized.append({"role": "user", "content": content_str})
        elif role == "assistant":
            # Assistant message завжди має content як рядок
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text = " ".join(text_parts)
                # Обрізати якщо занадто довге
                MAX_MESSAGE_LENGTH = 8000
                if len(text) > MAX_MESSAGE_LENGTH:
                    text = text[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                normalized.append({"role": "assistant", "content": text})
            elif isinstance(content, str):
                # Обрізати якщо занадто довге
                MAX_MESSAGE_LENGTH = 8000
                if len(content) > MAX_MESSAGE_LENGTH:
                    content = content[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                normalized.append({"role": "assistant", "content": content})
            else:
                content_str = str(content) if content else ""
                MAX_MESSAGE_LENGTH = 8000
                if len(content_str) > MAX_MESSAGE_LENGTH:
                    content_str = content_str[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                normalized.append({"role": "assistant", "content": content_str})
        elif role == "tool":
            # Tool message - обрізати результат якщо занадто великий
            MAX_MESSAGE_LENGTH = 8000
            tool_msg = msg.copy()
            if "content" in tool_msg:
                if isinstance(tool_msg["content"], list):
                    text_parts = []
                    for item in tool_msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content_str = " ".join(text_parts) if text_parts else ""
                    if len(content_str) > MAX_MESSAGE_LENGTH:
                        content_str = content_str[:MAX_MESSAGE_LENGTH] + "... [обрізано]"
                    tool_msg["content"] = content_str
                elif isinstance(tool_msg["content"], str):
                    if len(tool_msg["content"]) > MAX_MESSAGE_LENGTH:
                        tool_msg["content"] = tool_msg["content"][:MAX_MESSAGE_LENGTH] + "... [обрізано]"
            normalized.append(tool_msg)
    
    return normalized


def select_model(message: str, settings: dict, use_assistants: bool = False) -> str:
    """
    Автоматичний вибір моделі на основі складності запиту
    
    Правила:
    - gpt-4o-mini: прості запити, короткі відповіді
    - gpt-4o: складні запити, багатокрокові задачі, аналіз
    """
    if USE_LM_STUDIO:
        return "local-model"
    
    # Отримати модель з налаштувань
    user_model = settings.get("model", "gpt-4o-mini")
    
    # Якщо встановлено "auto", визначити автоматично
    if user_model == "auto":
        # Продовжити до автоматичного визначення
        pass
    # Якщо користувач явно вказав модель, використати її
    elif user_model in ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]:
        return user_model
    
    # Визначити складність запиту
    message_lower = message.lower()
    complexity_indicators = [
        "analyze", "compare", "explain", "why", "how", "what is",
        "проаналізуй", "порівняй", "поясни", "чому", "як", "що таке",
        "multiple", "several", "many", "кілька", "декілька",
        "complex", "detailed", "складний", "детальний",
        "calculate", "розрахуй", "порахуй",
        "and", "також", "також", "потім", "then"
    ]
    
    # Перевірити довжину запиту
    is_long = len(message.split()) > 20
    
    # Перевірити наявність індикаторів складності
    has_complexity = any(indicator in message_lower for indicator in complexity_indicators)
    
    # Якщо використовуємо Assistants API, завжди використовуємо gpt-4o для кращої якості
    if use_assistants:
        return "gpt-4o"
    
    # Вибрати модель
    if has_complexity or is_long:
        return "gpt-4o"
    else:
        return "gpt-4o-mini"


# ==================== ASSISTANTS API FUNCTIONS ====================
async def get_or_create_assistant(thread_id: str, settings: dict, vector_store_id: Optional[str] = None) -> Optional[str]:
    """Створити або отримати Assistant для thread"""
    if not client or USE_LM_STUDIO:
        return None
    
    # Перевірити кеш
    if thread_id in assistants_cache:
        return assistants_cache[thread_id].get("assistant_id")
    
    try:
        # Отримати увімкнені тули
        enabled_tools = get_enabled_tools(settings)
        
        # Визначити модель
        model = select_model("", settings, use_assistants=True)
        
        # System prompt
        system_prompt = """You are an AI assistant with access to tools. You MUST use tools when user asks you to perform actions.

CRITICAL RULES:
1. If user says "send email", "надішли листа", "відправити email" - IMMEDIATELY call send_email tool
2. If user says "book meeting", "забронювати", "schedule" - IMMEDIATELY call book_meeting tool
3. If user asks about price - call get_item_price tool
4. If user asks about shipping - call calculate_shipping tool

DO NOT:
- Say "I cannot send emails" - you CAN and MUST use send_email tool
- Just write the email text without sending - you MUST call send_email
- Ask for confirmation - just do it if user clearly asked

When user asks you to send an email, extract:
- recipient from the current or previous messages
- subject (create appropriate one if not specified, use context from conversation)
- body (use the email content from PREVIOUS conversation messages if user said "send that email" or "надішли того листа", otherwise create appropriate body)

IMPORTANT: If user says "send that email" or "надішли того листа", look in the conversation history for the email content that was written earlier. Extract the full email text from previous assistant messages and use it as the body.

Then IMMEDIATELY call send_email tool. Do not ask questions - just do it.

If user mentions an email address in the conversation, remember it and use it when they ask to send email."""
        
        # Підготувати tools для Assistant
        tools = []
        for tool_schema in enabled_tools:
            tools.append({
                "type": "function",
                "function": tool_schema["function"]
            })
        
        # Додати file_search якщо є vector store
        tool_resources = {}
        if vector_store_id:
            tools.append({"type": "file_search"})
            tool_resources = {
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        
        # Створити Assistant
        assistant = client.beta.assistants.create(
            name=f"Enterprise Assistant {thread_id}",
            instructions=system_prompt,
            model=model,
            tools=tools,
            tool_resources=tool_resources if tool_resources else None,
            temperature=settings.get("temperature", 0.7),
        )
        
        # Зберегти в кеш
        assistants_cache[thread_id] = {
            "assistant_id": assistant.id,
            "openai_thread_id": None
        }
        
        return assistant.id
    except Exception as e:
        print(f"⚠️  Помилка створення Assistant: {e}")
        return None


async def get_or_create_thread(thread_id: str) -> Optional[str]:
    """Створити або отримати OpenAI Thread"""
    if not client or USE_LM_STUDIO:
        return None
    
    # Перевірити кеш
    if thread_id in assistants_cache and assistants_cache[thread_id].get("openai_thread_id"):
        return assistants_cache[thread_id]["openai_thread_id"]
    
    try:
        # Створити новий Thread
        thread = client.beta.threads.create()
        
        # Зберегти в кеш
        if thread_id not in assistants_cache:
            assistants_cache[thread_id] = {}
        assistants_cache[thread_id]["openai_thread_id"] = thread.id
        
        return thread.id
    except Exception as e:
        print(f"⚠️  Помилка створення Thread: {e}")
        return None


async def handle_assistant_tool_calls(run, openai_thread_id: str, assistant_id: str) -> List[dict]:
    """Обробити tool calls від Assistant"""
    tools_used = []
    
    if run.required_action and run.required_action.type == "submit_tool_outputs":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        
        tool_outputs = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            func = available_functions.get(func_name)
            if func:
                try:
                    # Викликати функцію
                    if func_name == "get_item_price":
                        result = func(args.get("item_name", ""))
                    elif func_name == "calculate_shipping":
                        result = func(args.get("destination", ""), args.get("price", 0))
                    elif func_name == "book_meeting":
                        result = func(
                            args.get("topic", ""),
                            args.get("datetime_str", ""),
                            args.get("participants", "")
                        )
                    elif func_name == "send_email":
                        result = func(
                            args.get("recipient", ""),
                            args.get("subject", ""),
                            args.get("body", "")
                        )
                    else:
                        result = func(**args)
                    
                    tools_used.append({
                        "type": "tool",
                        "name": func_name,
                        "result": str(result),
                    })
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": str(result)
                    })
                except Exception as e:
                    error_result = json.dumps({"error": str(e)})
                    tools_used.append({
                        "type": "tool",
                        "name": func_name,
                        "result": error_result,
                    })
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": error_result
                    })
        
        # Відправити результати назад
        if tool_outputs:
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=openai_thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
    
    return tools_used

# ==================== IMAGE FUNCTIONS ====================
# Галерея збережених генерацій
GALLERY_FILE = "gallery.json"

def load_gallery():
    """Завантажити галерею з файлу"""
    global image_gallery
    if os.path.exists(GALLERY_FILE):
        try:
            with open(GALLERY_FILE, "r", encoding="utf-8") as f:
                image_gallery = json.load(f)
            print(f"✅ Завантажено {len(image_gallery)} зображень з галереї")
        except Exception as e:
            print(f"⚠️  Помилка завантаження галереї: {e}")
            image_gallery = []
    else:
        image_gallery = []

def save_gallery():
    """Зберегти галерею в файл"""
    try:
        with open(GALLERY_FILE, "w", encoding="utf-8") as f:
            json.dump(image_gallery, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  Помилка збереження галереї: {e}")

# Завантажити галерею при старті
image_gallery = []
load_gallery()

def generate_image(prompt: str, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard", style: str = "vivid"):
    """Генерація зображення через DALL-E API"""
    if not client or USE_LM_STUDIO:
        return None, "Помилка: OpenAI клієнт не ініціалізовано або використовується LM Studio. DALL-E потребує реального OpenAI API."
    
    try:
        # Визначити модель DALL-E
        dall_e_model = "dall-e-3" if model in ["dall-e-3", "dall-e"] else "dall-e-2"
        
        # Параметри для DALL-E-3
        if dall_e_model == "dall-e-3":
            # DALL-E-3 підтримує тільки 1024x1024, 1792x1024, 1024x1792
            valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
            if size not in valid_sizes:
                size = "1024x1024"
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,  # "standard" або "hd"
                style=style,  # "vivid" або "natural"
                n=1,
            )
        else:
            # DALL-E-2
            valid_sizes = ["256x256", "512x512", "1024x1024"]
            if size not in valid_sizes:
                size = "1024x1024"
            
            response = client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size=size,
                n=1,
            )
        
        image_url = response.data[0].url
        
        # Зберегти в галерею
        # Генерувати унікальний ID на основі timestamp
        new_id = int(datetime.now().timestamp() * 1000)  # Мілісекунди для унікальності
        gallery_item = {
            "id": new_id,
            "prompt": prompt,
            "image_url": image_url,
            "timestamp": datetime.now().isoformat(),
            "model": dall_e_model,
            "size": size,
            "quality": quality if dall_e_model == "dall-e-3" else None,
            "style": style if dall_e_model == "dall-e-3" else None,
        }
        image_gallery.append(gallery_item)
        save_gallery()  # Зберегти в файл
        
        return image_url, None
    except Exception as e:
        error_msg = f"Помилка генерації зображення: {str(e)}"
        print(f"⚠️  {error_msg}")
        return None, error_msg


def analyze_image(image_base64: str, question: str = None, detailed: bool = True):
    """Аналіз зображення через GPT-4V (VQA - Visual Question Answering)"""
    if not client:
        return "Помилка: OpenAI клієнт не ініціалізовано. Переконайтеся, що API ключ встановлено або LM Studio запущено."
    
    # Використовувати GPT-4o для vision (найкраща якість)
    model = "gpt-4o"
    
    # Детальний промпт для VQA
    if question:
        vqa_prompt = f"""Analyze this image and answer the following question in detail: {question}

Provide a comprehensive answer that includes:
- Direct answer to the question
- Relevant details from the image
- Context and observations that support your answer"""
    else:
        vqa_prompt = """Analyze this image in detail. Provide a comprehensive description that includes:
- Main subjects and objects
- Colors, composition, and visual style
- Text or symbols if present
- Mood, atmosphere, or emotional tone
- Any notable details or interesting elements
- Potential context or meaning"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": vqa_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high" if detailed else "low"  # high для детального аналізу
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000 if detailed else 300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Помилка аналізу зображення: {str(e)}"


# ==================== MAIN CHAT LOGIC ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Головний endpoint для обробки запитів"""

    response_content = ""
    tools_used = []
    image_url = None

    # ===== MODE: IMAGE GENERATION =====
    if request.mode == "image-gen":
        image_settings = request.settings.get("imageSettings", {})
        image_url, error = generate_image(
            prompt=request.message,
            model=image_settings.get("model", "dall-e-3"),
            size=image_settings.get("size", "1024x1024"),
            quality=image_settings.get("quality", "standard"),
            style=image_settings.get("style", "vivid")
        )
        if error:
            response_content = error
        else:
            response_content = f"Зображення згенеровано на основі запиту: {request.message}"
            tools_used.append({"type": "image", "url": image_url, "prompt": request.message})

    # ===== MODE: IMAGE ANALYSIS (VQA) =====
    elif request.mode == "image-analyze" and request.image_base64:
        # Використати повідомлення як питання для VQA
        question = request.message if request.message.strip() else None
        detailed = request.settings.get("detailedAnalysis", True)
        analysis = analyze_image(request.image_base64, question, detailed)
        response_content = analysis
        tools_used.append({
            "type": "vision", 
            "data": "Image analyzed",
            "question": question,
            "detailed": detailed
        })

    # ===== MODE: CHAT WITH RAG + AGENT =====
    else:
        thread_id = request.thread_id
        
        # Використовувати Assistants API якщо доступний OpenAI API (не LM Studio)
        # ТИМЧАСОВО ВИМКНЕНО через проблеми з Vector Stores API
        # Використовуємо Chat Completions API (більш стабільний)
        use_assistants_api = False
        # if client and not USE_LM_STUDIO and request.settings.get("enableAgent", True):
        #     try:
        #         # Перевірити чи підтримується Assistants API
        #         if hasattr(client, 'beta') and hasattr(client.beta, 'assistants'):
        #             use_assistants_api = True
        #         else:
        #             print("⚠️  Assistants API не доступний, використовується Chat Completions API")
        #     except Exception as e:
        #         print(f"⚠️  Помилка перевірки Assistants API: {e}, використовується Chat Completions API")
        
        if use_assistants_api:
            # ===== ASSISTANTS API PATH =====
            try:
                # Створити або отримати Vector Store для RAG
                vector_store_id = None
                if request.settings.get("enableRAG", True):
                    vector_store_id = await create_or_get_vector_store(thread_id)
                
                # Створити або отримати Assistant
                assistant_id = await get_or_create_assistant(thread_id, request.settings, vector_store_id)
                if not assistant_id:
                    raise Exception("Не вдалося створити Assistant")
                
                # Створити або отримати Thread
                openai_thread_id = await get_or_create_thread(thread_id)
                if not openai_thread_id:
                    raise Exception("Не вдалося створити Thread")
                
                # Додати повідомлення користувача до Thread
                client.beta.threads.messages.create(
                    thread_id=openai_thread_id,
                    role="user",
                    content=request.message
                )
                
                # Створити Run
                run = client.beta.threads.runs.create(
                    thread_id=openai_thread_id,
                    assistant_id=assistant_id
                )
                
                # Чекати завершення Run (з обробкою tool calls)
                max_iterations = 10
                iteration = 0
                while run.status in ["queued", "in_progress", "requires_action"] and iteration < max_iterations:
                    if run.status == "requires_action":
                        # Обробити tool calls
                        tool_results = await handle_assistant_tool_calls(run, openai_thread_id, assistant_id)
                        tools_used.extend(tool_results)
                        
                        # Оновити run
                        run = client.beta.threads.runs.retrieve(
                            thread_id=openai_thread_id,
                            run_id=run.id
                        )
                    
                    if run.status in ["queued", "in_progress"]:
                        import time
                        time.sleep(1)
                        run = client.beta.threads.runs.retrieve(
                            thread_id=openai_thread_id,
                            run_id=run.id
                        )
                    
                    iteration += 1
                
                # Отримати останні повідомлення
                messages = client.beta.threads.messages.list(
                    thread_id=openai_thread_id,
                    limit=1
                )
                
                if messages.data and messages.data[0].content:
                    response_content = ""
                    for content_block in messages.data[0].content:
                        if content_block.type == "text":
                            response_content += content_block.text.value
                    
                    # Додати RAG інформацію якщо використовувався file_search
                    if vector_store_id and tools_used:
                        tools_used.append({
                            "type": "rag",
                            "docs": ["file_search"],
                        })
                else:
                    response_content = "Не вдалося отримати відповідь від Assistant"
                    
            except Exception as e:
                print(f"⚠️  Помилка Assistants API: {e}")
                import traceback
                traceback.print_exc()
                # Fallback до старого методу
                use_assistants_api = False
        
        if not use_assistants_api:
            # ===== LEGACY PATH (Chat Completions API) =====
            # Отримати або ініціалізувати історію для цього thread
            if thread_id not in conversation_history:
                conversation_history[thread_id] = [
                    {
                        "role": "system",
                        "content": f"You are a smart enterprise assistant. Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}."
                    }
                ]
            
            # Побудувати історію повідомлень
            # Базовий system prompt (буде доповнений RAG контекстом якщо потрібно)
            system_prompt = """You are an AI assistant with access to tools and a knowledge base.

PRIORITY ORDER:
1. FIRST: Check if the user's question can be answered using information from the knowledge base (documents that were uploaded)
2. SECOND: If the question cannot be answered from documents, use tools for actions (send email, book meeting, etc.)
3. If user asks about information that should be in documents, ALWAYS check the knowledge base first before using tools

CRITICAL RULES FOR TOOLS:
- If user says "send email", "надішли листа", "відправити email" - IMMEDIATELY call send_email tool
- If user says "book meeting", "забронювати", "schedule" - IMMEDIATELY call book_meeting tool
- If user asks about price of items NOT in documents - call get_item_price tool
- If user asks about shipping - call calculate_shipping tool

DO NOT:
- Use tools for information that should be in the knowledge base documents
- Say "I cannot send emails" - you CAN and MUST use send_email tool
- Just write the email text without sending - you MUST call send_email
- Ask for confirmation - just do it if user clearly asked

When user asks you to send an email, extract:
- recipient from the current or previous messages
- subject (create appropriate one if not specified, use context from conversation)
- body (use the email content from PREVIOUS conversation messages if user said "send that email" or "надішли того листа", otherwise create appropriate body)

IMPORTANT: If user says "send that email" or "надішли того листа", look in the conversation history for the email content that was written earlier. Extract the full email text from previous assistant messages and use it as the body.

Then IMMEDIATELY call send_email tool. Do not ask questions - just do it.

If user mentions an email address in the conversation, remember it and use it when they ask to send email."""
            
            # Оновити system prompt в історії
            conversation_history[thread_id][0]["content"] = system_prompt
            
            # Додати поточне повідомлення користувача до історії
            user_message = {
                "role": "user",
            }
            
            # Обробити image_base64 якщо є
            if request.image_base64:
                user_message["content"] = [
                    {
                        "type": "text",
                        "text": request.message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{request.image_base64}"
                        }
                    }
                ]
            else:
                user_message["content"] = request.message
            
            conversation_history[thread_id].append(user_message)
            
            # Обмежити історію до останніх N повідомлень (щоб не перевищити ліміт токенів)
            MAX_HISTORY_MESSAGES = 30  # Максимум повідомлень в історії
            history = conversation_history[thread_id].copy()
            
            # Якщо історія занадто довга, залишити тільки system prompt + останні N повідомлень
            if len(history) > MAX_HISTORY_MESSAGES:
                # Зберегти system prompt (перший елемент)
                system_msg = history[0] if history and history[0].get("role") == "system" else None
                # Взяти останні N повідомлень (без system prompt)
                recent_messages = history[-(MAX_HISTORY_MESSAGES - 1):] if system_msg else history[-MAX_HISTORY_MESSAGES:]
                # Об'єднати system prompt + останні повідомлення
                if system_msg:
                    messages = [system_msg] + recent_messages
                else:
                    messages = recent_messages
                print(f"⚠️  Історія обрізана: {len(history)} -> {len(messages)} повідомлень")
            else:
                messages = history

            # RAG: Витягнути документи, якщо увімкнено
            enable_rag = request.settings.get("enableRAG", True)
            print(f"🔍 RAG enabled: {enable_rag}, mode: {request.mode}")
            
            has_rag_context = False  # Флаг що є RAG контекст
            rag_file_names = []
            
            if enable_rag:
                docs = retrieve_relevant_docs(request.message, n_results=5)
                print(f"📚 RAG retrieved {len(docs)} documents")
                
                if docs:
                    has_rag_context = True
                    # Обмежити розмір контексту (максимум 5000 символів)
                    MAX_CONTEXT_LENGTH = 5000
                    context_parts = []
                    file_names = []
                    total_length = 0
                    
                    for doc in docs:
                        doc_text = doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc)
                        source_name = doc.get("source", "unknown") if isinstance(doc, dict) else "unknown"
                        
                        # Додати назву файлу до списку
                        if source_name not in file_names:
                            file_names.append(source_name)
                        
                        if total_length + len(doc_text) > MAX_CONTEXT_LENGTH:
                            # Додати частину останнього документа
                            remaining = MAX_CONTEXT_LENGTH - total_length
                            if remaining > 100:  # Якщо залишилося достатньо місця
                                context_parts.append(f"[From {source_name}]\n{doc_text[:remaining]}...")
                            break
                        context_parts.append(f"[From {source_name}]\n{doc_text}")
                        total_length += len(doc_text) + len(source_name) + 10
                    
                    context = "\n\n".join(context_parts)
                    
                    # Покращений system prompt з інструкціями використання контексту
                    rag_instruction = f"""

═══════════════════════════════════════════════════════════════
KNOWLEDGE BASE CONTEXT - USE THIS INFORMATION FIRST!
═══════════════════════════════════════════════════════════════

You have access to relevant documents from the user's knowledge base. The user's question MUST be answered using information from these documents if possible.

Relevant documents from knowledge base:
{context}

CRITICAL INSTRUCTIONS:
1. FIRST PRIORITY: Answer the user's question using ONLY the information from the documents above
2. DO NOT use tools (get_item_price, calculate_shipping, etc.) if the information is in the documents
3. If the information is in the documents, cite which document(s) you used (e.g., "According to [filename]...")
4. Be specific and accurate when referencing information from the documents
5. If the documents don't contain the information, say so clearly: "I cannot find this information in the uploaded documents"
6. ONLY use tools if the user explicitly asks for an ACTION (send email, book meeting) or if the information is NOT in the documents

EXAMPLE:
- User: "What is the goal of the lab work?"
- You: Check the documents first. If found, answer from documents. DO NOT call get_item_price tool.
- User: "Send an email to..."
- You: Use send_email tool (this is an action, not information retrieval)

═══════════════════════════════════════════════════════════════"""
                    
                    # Переконатися що system prompt має content як рядок
                    if isinstance(messages[0].get("content"), str):
                        messages[0]["content"] += rag_instruction
                    else:
                        # Якщо system prompt має інший формат, створити новий
                        messages[0] = {
                            "role": "system",
                            "content": f"{system_prompt}{rag_instruction}"
                        }
                    
                    # Зберегти назви файлів для візуалізації
                    rag_file_names = file_names if file_names else [f"doc_{i}" for i in range(len(docs))]
                    
                    # Передати реальні назви файлів для візуалізації
                    tools_used.append(
                        {
                            "type": "rag",
                            "docs": rag_file_names,
                        }
                    )
                    
                    print(f"📄 RAG files used: {rag_file_names}")
                else:
                    print("⚠️  RAG enabled but no documents found in knowledge base")

            # Нормалізувати messages перед відправкою до API
            messages = normalize_messages(messages, system_prompt)
            
            # Виклик LLM з інструментами
            if request.settings.get("enableAgent", True):
                if not client:
                    error_msg = "Помилка: OpenAI клієнт не ініціалізовано. "
                    if USE_LM_STUDIO:
                        error_msg += "Переконайтеся, що LM Studio запущено на http://localhost:1234"
                    else:
                        error_msg += "Переконайтеся, що OPENAI_API_KEY встановлено в .env файлі"
                    response_content = error_msg
                    return ChatResponse(
                        content=response_content, tools=tools_used, image_url=image_url
                    )
            
            # Отримати увімкнені тули
            # Якщо є RAG контекст, вимкнути інформаційні tools (get_item_price, calculate_shipping)
            # але залишити action tools (send_email, book_meeting)
            enabled_tools = get_enabled_tools(request.settings)
            
            if has_rag_context and enabled_tools:
                # Фільтрувати tools - залишити тільки action tools
                action_tools = ["send_email", "book_meeting"]
                enabled_tools = [
                    tool for tool in enabled_tools 
                    if tool["function"]["name"] in action_tools
                ]
                if enabled_tools:
                    print(f"🔧 RAG context found - only action tools enabled: {[t['function']['name'] for t in enabled_tools]}")
                else:
                    print(f"🔧 RAG context found - all tools disabled for information queries")
                    enabled_tools = None
            
            # Визначити модель залежно від складності запиту
            default_model = select_model(request.message, request.settings, use_assistants=False)
            
            # Логування для дебагу
            print(f"🔧 Enabled tools: {[t['function']['name'] for t in enabled_tools] if enabled_tools else 'None'}")
            print(f"📝 User message: {request.message[:100]}...")
            print(f"📊 Історія перед запитом: {len(messages)} повідомлень")
            if len(messages) > 0:
                print(f"   System prompt type: {type(messages[0].get('content'))}")
            if len(messages) > 1:
                print(f"   Останні 3 повідомлення:")
                for m in messages[-3:]:
                    role = m.get("role", "unknown")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content_preview = f"[array with {len(content)} items]"
                    else:
                        content_preview = str(content)[:50] if content else ""
                    tool_calls = m.get("tool_calls", [])
                    print(f"     - {role}: {content_preview}... {'[tool_calls]' if tool_calls else ''}")
            
            try:
                response = client.chat.completions.create(
                    model=default_model,
                    messages=messages,
                    tools=enabled_tools if enabled_tools else None,
                    temperature=request.settings.get("temperature", 0.7),
                )
            except Exception as e:
                error_str = str(e)
                # Обробка помилки 429 (Rate Limit / Too Many Tokens)
                if "429" in error_str or "rate_limit" in error_str.lower() or "too many requests" in error_str.lower() or "tokens per min" in error_str.lower():
                    # Спробувати зменшити історію та повторити
                    print(f"⚠️  Rate limit / Token limit error, спроба зменшити історію...")
                    # Обрізати історію до останніх 10 повідомлень
                    if len(messages) > 10:
                        system_msg = messages[0] if messages[0].get("role") == "system" else None
                        recent = messages[-(10 - (1 if system_msg else 0)):]
                        messages = [system_msg] + recent if system_msg else recent
                        print(f"   Зменшено до {len(messages)} повідомлень, повторна спроба...")
                        try:
                            response = client.chat.completions.create(
                                model=default_model,
                                messages=messages,
                                tools=enabled_tools if enabled_tools else None,
                                temperature=request.settings.get("temperature", 0.7),
                            )
                        except Exception as e2:
                            # Якщо все одно не працює, повернути помилку
                            error_msg = f"Помилка API: Запит занадто великий або перевищено ліміт запитів. Спробуйте скоротити повідомлення або зачекати."
                            print(f"❌ {error_msg}")
                            return ChatResponse(
                                content=error_msg,
                                tools=[],
                                image_url=None
                            )
                else:
                    # Інші помилки
                    error_msg = f"Помилка OpenAI API: {error_str}"
                    print(f"❌ {error_msg}")
                    return ChatResponse(
                        content=error_msg,
                        tools=[],
                        image_url=None
                    )

            msg = response.choices[0].message
            
            # Логування відповіді
            print(f"📊 Історія thread {thread_id}: {len(conversation_history.get(thread_id, []))} повідомлень")
            if msg.tool_calls:
                print(f"✅ AI викликає тули: {[tc.function.name for tc in msg.tool_calls]}")
            else:
                print(f"⚠️  AI не викликав тули. Відповідь: {msg.content[:100] if msg.content else 'None'}...")

            # Якщо LLM хоче викликати інструмент
            if msg.tool_calls:
                # Додати assistant message з tool_calls до messages тільки один раз
                assistant_msg_dict = {
                    "role": "assistant",
                    "content": msg.content if msg.content else None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in msg.tool_calls
                    ]
                }
                messages.append(assistant_msg_dict)
                
                # Додати assistant message до conversation_history тільки один раз
                conversation_history[thread_id].append(assistant_msg_dict)
                
                # Виконати всі тули
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    func = available_functions.get(func_name)
                    if func:
                        # Викликати функцію з правильними аргументами
                        try:
                            if func_name == "get_item_price":
                                result = func(args.get("item_name", ""))
                            elif func_name == "calculate_shipping":
                                result = func(args.get("destination", ""), args.get("price", 0))
                            elif func_name == "book_meeting":
                                result = func(
                                    args.get("topic", ""),
                                    args.get("datetime_str", ""),
                                    args.get("participants", "")
                                )
                            elif func_name == "send_email":
                                result = func(
                                    args.get("recipient", ""),
                                    args.get("subject", ""),
                                    args.get("body", "")
                                )
                            else:
                                result = func(**args)
                            
                            tools_used.append(
                                {
                                    "type": "tool",
                                    "name": func_name,
                                    "result": str(result),
                                }
                            )
                        except Exception as e:
                            result = json.dumps({"error": str(e)})
                            tools_used.append(
                                {
                                    "type": "tool",
                                    "name": func_name,
                                    "result": result,
                                }
                            )

                        # Створити tool response
                        tool_response = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": str(result),  # Вже JSON string
                        }
                        # Додати tool response до messages
                        messages.append(tool_response)
                        
                        # Додати tool response до conversation_history
                        conversation_history[thread_id].append(tool_response)

                # Фінальна відповідь після виконання інструментів
                final_model = select_model(request.message, request.settings, use_assistants=False)
                
                final_response = client.chat.completions.create(
                    model=final_model,
                    messages=messages,
                )
                response_content = final_response.choices[0].message.content
                
                # Додати фінальну відповідь до історії
                conversation_history[thread_id].append({
                    "role": "assistant",
                    "content": response_content
                })
            else:
                response_content = msg.content
                # Додати відповідь до історії
                conversation_history[thread_id].append({
                    "role": "assistant",
                    "content": response_content
                })
        else:
            # Без агента - проста LLM відповідь
            # Визначити system_prompt для цього блоку
            simple_system_prompt = f"You are a smart enterprise assistant. Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}."
            
            if not client:
                error_msg = "Помилка: OpenAI клієнт не ініціалізовано. "
                if USE_LM_STUDIO:
                    error_msg += "Переконайтеся, що LM Studio запущено на http://localhost:1234"
                else:
                    error_msg += "Переконайтеся, що OPENAI_API_KEY встановлено в .env файлі"
                response_content = error_msg
            else:
                model = select_model(request.message, request.settings, use_assistants=False)
                
                # Нормалізувати messages перед відправкою до API
                messages = normalize_messages(messages, simple_system_prompt)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=request.settings.get("temperature", 0.7),
                )
                response_content = response.choices[0].message.content
                
                # Додати відповідь до історії
                conversation_history[thread_id].append({
                    "role": "assistant",
                    "content": response_content
                })

    return ChatResponse(
        content=response_content, tools=tools_used, image_url=image_url
    )


# ==================== STREAMING ENDPOINT ====================
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint для chat (тільки для Chat Completions API, не Assistants API)"""
    
    if request.mode != "chat" or not request.settings.get("enableAgent", True):
        # Streaming не підтримується для image modes або без agent
        return {"error": "Streaming доступний тільки для chat mode з agent"}
    
    if not client or USE_LM_STUDIO:
        return {"error": "Streaming доступний тільки з OpenAI API"}
    
    thread_id = request.thread_id
    
    # Ініціалізувати історію якщо потрібно
    if thread_id not in conversation_history:
        conversation_history[thread_id] = [
            {
                "role": "system",
                "content": "You are an AI assistant with access to tools."
            }
        ]
    
    # Додати повідомлення користувача
    conversation_history[thread_id].append({
        "role": "user",
        "content": request.message
    })
    
    messages = conversation_history[thread_id].copy()
    
    # Отримати увімкнені тули
    enabled_tools = get_enabled_tools(request.settings)
    
    # Визначити модель
    model = select_model(request.message, request.settings, use_assistants=False)
    
    async def generate():
        """Async generator для streaming з правильним flush"""
        try:
            # Створити streaming request
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=enabled_tools if enabled_tools else None,
                temperature=request.settings.get("temperature", 0.7),
                stream=True,
            )
            
            full_content = ""
            has_content = False
            tool_calls_accumulated = []
            
            for chunk in stream:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                    
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                
                # Обробка текстового контенту
                if delta.content:
                    content = delta.content
                    full_content += content
                    has_content = True
                    # Відправити chunk одразу
                    data = json.dumps({'content': content, 'done': False})
                    yield f"data: {data}\n\n"
                    # Дати можливість event loop обробити інші завдання
                    await asyncio.sleep(0)
                
                # Обробка tool calls (якщо є)
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        idx = tool_call_delta.index
                        
                        # Ініціалізувати tool call якщо потрібно
                        while len(tool_calls_accumulated) <= idx:
                            tool_calls_accumulated.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        # Оновити tool call
                        if tool_call_delta.id:
                            tool_calls_accumulated[idx]["id"] = tool_call_delta.id
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tool_calls_accumulated[idx]["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tool_calls_accumulated[idx]["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Якщо є tool calls, завершити streaming та повернутися до non-streaming
            if tool_calls_accumulated and any(tc.get("function", {}).get("name") for tc in tool_calls_accumulated):
                # Streaming не підтримує tool calls добре, тому завершуємо streaming
                data = json.dumps({
                    'content': '', 
                    'done': True, 
                    'full_content': full_content, 
                    'has_tools': True, 
                    'message': 'Tool calls detected, switching to non-streaming mode'
                })
                yield f"data: {data}\n\n"
                return
            
            # Зберегти повну відповідь в історію
            if has_content and full_content:
                conversation_history[thread_id].append({
                    "role": "assistant",
                    "content": full_content
                })
            
            # Відправити фінальний сигнал
            data = json.dumps({'content': '', 'done': True, 'full_content': full_content})
            yield f"data: {data}\n\n"
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Streaming error: {error_msg}")
            import traceback
            traceback.print_exc()
            data = json.dumps({'error': error_msg, 'done': True})
            yield f"data: {data}\n\n"
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Вимкнути буферизацію в nginx
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )


# ==================== RAG ENDPOINTS ====================
@app.post("/upload_documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    thread_id: Optional[str] = Form(None)
):
    """Завантажити документи у RAG базу (OpenAI File Search API або ChromaDB)"""
    print(f"📤 Завантаження {len(files)} файлів, thread_id: {thread_id}")
    uploaded_files = []
    file_ids = []
    file_contents = {}  # Зберігати контент для fallback (визначено тут для використання в fallback)
    
    # Спробувати завантажити в OpenAI File Search API
    if client and not USE_LM_STUDIO:
        try:
            # Створити або отримати Vector Store
            if thread_id:
                vector_store_id = await create_or_get_vector_store(thread_id)
            else:
                # Створити глобальний Vector Store
                vector_store_id = await create_or_get_vector_store("global")
            
            if vector_store_id:
                errors = []
                for file in files:
                    try:
                        content = await file.read()
                        file_contents[file.filename] = content  # Зберегти для fallback
                        file_id = await upload_file_to_openai(content, file.filename)
                        if file_id:
                            await add_file_to_vector_store(file_id, vector_store_id)
                            file_ids.append(file_id)
                            uploaded_files.append(file.filename)
                        else:
                            errors.append(f"{file.filename}: не вдалося завантажити в OpenAI")
                    except Exception as e:
                        error_msg = f"{file.filename}: {str(e)}"
                        errors.append(error_msg)
                        print(f"⚠️  Помилка завантаження файлу {file.filename}: {e}")
                
                if file_ids:
                    result = {
                        "status": "success",
                        "count": len(file_ids),
                        "method": "openai_file_search",
                        "files": uploaded_files
                    }
                    if errors:
                        result["warnings"] = errors
                    return result
                else:
                    # Якщо жоден файл не завантажився, продовжити до ChromaDB fallback
                    print(f"⚠️  Не вдалося завантажити файли в OpenAI: {errors}")
        except Exception as e:
            print(f"⚠️  Помилка завантаження в OpenAI File Search: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback до ChromaDB (legacy)
    docs = []
    errors = []
    
    # Використати збережений контент або прочитати заново
    for file in files:
        try:
            # Використати збережений контент якщо є, інакше прочитати
            if file.filename in file_contents:
                content = file_contents[file.filename]
            else:
                # Спробувати повернути файл на початок
                try:
                    await file.seek(0)
                    content = await file.read()
                except Exception as seek_error:
                    # Якщо seek не працює, спробувати прочитати безпосередньо
                    print(f"⚠️  Не вдалося seek для {file.filename}: {seek_error}")
                    # Для FastAPI UploadFile, якщо вже прочитано, потрібно використати збережений контент
                    content = b""  # Порожній контент, файл буде пропущено
            
            # Перевірити що контент не порожній
            if not content or len(content) == 0:
                errors.append(f"{file.filename}: файл порожній або не вдалося прочитати")
                continue
            
            # Спробувати декодувати як текст
            try:
                text = content.decode("utf-8")
                if text.strip():  # Перевірити що файл не порожній
                    docs.append({"text": text, "source": file.filename})
                else:
                    errors.append(f"{file.filename}: файл порожній після декодування")
            except UnicodeDecodeError:
                # Якщо не UTF-8, спробувати інші кодування або пропустити
                try:
                    text = content.decode("latin-1")
                    if text.strip():
                        docs.append({"text": text, "source": file.filename})
                    else:
                        errors.append(f"{file.filename}: не вдалося декодувати як текст")
                except:
                    errors.append(f"{file.filename}: не текстовий файл (потрібен PDF/DOC парсер)")
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if docs:
        try:
            add_documents_to_rag(docs)
            result = {
                "status": "success",
                "count": len(docs),
                "method": "chromadb",
                "files": [doc["source"] for doc in docs]
            }
            if errors:
                result["warnings"] = errors
            return result
        except Exception as e:
            errors.append(f"ChromaDB error: {str(e)}")
    
    # Якщо нічого не вдалося завантажити
    error_message = "Не вдалося завантажити файли"
    if errors:
        error_message += f". Помилки: {', '.join(errors)}"
    elif not CHROMADB_AVAILABLE:
        error_message += ". ChromaDB не встановлено. Встановіть: pip install chromadb"
    elif not client:
        error_message += ". OpenAI клієнт не ініціалізовано"
    elif USE_LM_STUDIO:
        error_message += ". OpenAI File Search недоступний при використанні LM Studio"
    
    print(f"❌ Помилка завантаження: {error_message}")
    print(f"   Деталі: errors={errors}, CHROMADB_AVAILABLE={CHROMADB_AVAILABLE}, client={client is not None}, USE_LM_STUDIO={USE_LM_STUDIO}")
    return {"status": "error", "message": error_message, "errors": errors}


@app.get("/search_documents")
async def search_documents(query: str):
    """Пошук у RAG базі"""
    results = retrieve_relevant_docs(query)
    return {"results": results}


# ==================== CATALOG ENDPOINTS ====================
@app.post("/upload_catalog")
async def upload_catalog(file: UploadFile = File(...)):
    """Завантажити CSV каталог товарів"""
    global PRODUCT_CATALOG
    try:
        content = await file.read()
        # Зберегти тимчасово для pandas
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        df = pd.read_csv(tmp_path)
        new_catalog = {}
        for _, row in df.iterrows():
            item_name = str(row.get('item_name', '')).strip().lower()
            price = int(row.get('price', 0))
            if item_name and price:
                new_catalog[item_name] = price
        
        PRODUCT_CATALOG = new_catalog
        os.unlink(tmp_path)  # Видалити тимчасовий файл
        
        return {"status": "success", "count": len(new_catalog), "message": f"Завантажено {len(new_catalog)} товарів"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/catalog")
async def get_catalog():
    """Отримати поточний каталог"""
    return {"catalog": PRODUCT_CATALOG, "count": len(PRODUCT_CATALOG)}


# ==================== GOOGLE AUTH ENDPOINTS ====================
@app.post("/google_auth")
async def google_auth():
    """Ініціалізувати Google авторизацію"""
    if not GOOGLE_AVAILABLE:
        return {"error": "Google API бібліотеки не встановлено"}
    
    if not os.path.exists(CREDENTIALS_PATH):
        return {"error": f"Файл credentials.json не знайдено за адресою: {CREDENTIALS_PATH}"}
    
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        # Для веб-додатку потрібно повернути URL для авторизації
        # Але для простоти повертаємо інструкції
        return {
            "status": "info",
            "message": "Для авторизації запустіть скрипт auth_google.py або використайте OAuth flow"
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== IMAGE ENDPOINTS ====================
@app.post("/generate_image")
async def generate_image_endpoint(
    prompt: str, 
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid"
):
    """Ендпоінт для генерації зображень через DALL-E API"""
    image_url, error = generate_image(prompt, model, size, quality, style)
    if error:
        return {"error": error, "image_url": None}
    return {
        "image_url": image_url,
        "prompt": prompt,
        "model": model,
        "size": size
    }


@app.post("/analyze_image")
async def analyze_image_endpoint(
    file: UploadFile = File(...), 
    question: Optional[str] = None,
    detailed: bool = True
):
    """Ендпоінт для аналізу зображень (VQA)"""
    content = await file.read()
    image_base64 = base64.b64encode(content).decode("utf-8")
    analysis = analyze_image(image_base64, question, detailed)
    return {"analysis": analysis, "question": question}


@app.get("/gallery")
async def get_gallery(limit: int = 50):
    """Отримати галерею згенерованих зображень"""
    return {
        "gallery": image_gallery[-limit:][::-1],  # Останні N, від новіших до старіших
        "total": len(image_gallery)
    }


@app.delete("/gallery/{image_id}")
async def delete_from_gallery(image_id: int):
    """Видалити зображення з галереї"""
    global image_gallery
    image_gallery = [img for img in image_gallery if img["id"] != image_id]
    save_gallery()  # Зберегти зміни
    return {"status": "deleted", "remaining": len(image_gallery)}


@app.delete("/gallery")
async def clear_gallery():
    """Очистити всю галерею"""
    global image_gallery
    count = len(image_gallery)
    image_gallery = []
    save_gallery()  # Зберегти зміни
    return {"status": "cleared", "deleted_count": count}


# ==================== HISTORY ENDPOINTS ====================
@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Отримати історію розмови для thread"""
    if thread_id in conversation_history:
        return {"history": conversation_history[thread_id], "count": len(conversation_history[thread_id])}
    return {"history": [], "count": 0}


@app.delete("/history/{thread_id}")
async def clear_history(thread_id: str):
    """Очистити історію розмови для thread"""
    if thread_id in conversation_history:
        # Залишити тільки system prompt
        system_msg = conversation_history[thread_id][0] if conversation_history[thread_id] else None
        conversation_history[thread_id] = [system_msg] if system_msg else []
        return {"status": "cleared"}
    return {"status": "not_found"}


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn

    # Запуск з параметрами для правильного streaming
    # --no-access-log: вимкнути логи для швидшої роботи
    # --timeout-keep-alive: збільшити timeout для streaming
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 хвилин для streaming
        log_level="info"
    )

