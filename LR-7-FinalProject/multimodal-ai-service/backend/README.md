# Backend для Multimodal AI Service

FastAPI backend сервер, який поєднує:

- **RAG** (Retrieval-Augmented Generation) - векторний пошук документів
- **Agent Tools** - виконання функцій через LLM
- **Image Generation** - генерація зображень
- **Image Analysis** - аналіз зображень через vision моделі

## Встановлення

### Базове встановлення (без RAG)

```bash
cd backend
pip install -r requirements-base.txt
```

### Повне встановлення (з RAG через ChromaDB)

```bash
cd backend
pip install -r requirements.txt
```

**Примітка для Windows:** Якщо встановлення `chromadb` не вдається через помилку з Visual C++, у вас є два варіанти:

1. **Встановити Visual C++ Build Tools:**

   - Завантажте з: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Встановіть "Desktop development with C++"
   - Після цього виконайте: `pip install chromadb`

2. **Використати базову версію без RAG:**
   - Використайте `requirements-base.txt` - сервер працюватиме, але RAG буде недоступний

## Налаштування

Backend підтримує два режими роботи:

### Режим 1: Реальний OpenAI API (за замовчуванням)

1. Створіть файл `.env` в папці `backend/`:

```bash
OPENAI_API_KEY=your-api-key-here
USE_LM_STUDIO=false
```

2. Або встановіть змінну оточення:

```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Або встановіть API ключ безпосередньо в `main.py` (рядок 40) для швидкого тестування

### Режим 2: LM Studio (локальна модель)

1. Встановіть в `.env`:

```bash
USE_LM_STUDIO=true
LM_STUDIO_URL=http://localhost:1234/v1
```

2. Запустіть LM Studio на `http://localhost:1234`
3. Завантажте модель (наприклад, LLaVA для vision або будь-яку іншу для чату)

## Запуск

```bash
python main.py
```

Або через uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST `/chat`

Головний endpoint для обробки чат запитів.

**Request:**

```json
{
  "thread_id": "123",
  "message": "What is the price of iPhone 15?",
  "mode": "chat",
  "image_base64": null,
  "settings": {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "enableRAG": true,
    "enableAgent": true
  }
}
```

**Response:**

```json
{
  "content": "The price of iPhone 15 is $900",
  "tools": [
    {
      "type": "tool",
      "name": "get_item_price",
      "result": "{'item': 'iphone 15', 'price': 900}"
    }
  ],
  "image_url": null
}
```

### POST `/upload_documents`

Завантажити документи у RAG базу даних.

### GET `/search_documents?query=...`

Пошук документів у RAG базі.

### POST `/generate_image`

Генерація зображення (поки mock).

### POST `/analyze_image`

Аналіз завантаженого зображення.

## Структура

- `main.py` - головний файл з усіма endpoints
- `chroma_db/` - векторна база даних (створюється автоматично)
- `requirements.txt` - залежності Python

## Agent Tools

Backend підтримує 4 тули:

1. **get_item_price** - отримати ціну товару з каталогу
2. **calculate_shipping** - розрахувати вартість доставки
3. **book_meeting** - забронювати зустріч через Google Calendar
4. **send_email** - відправити email через Gmail

### Налаштування Google API

1. Створіть проект в [Google Cloud Console](https://console.cloud.google.com/)
2. Увімкніть Calendar API та Gmail API
3. Створіть OAuth 2.0 credentials
4. Завантажте `credentials.json` в папку `backend/`
5. Запустіть авторизацію:

```bash
python auth_google.py
```

6. Після авторизації буде створено `token.json`

### Завантаження каталогу товарів

Завантажте CSV файл з колонками `item_name` та `price`:

```bash
curl -X POST "http://localhost:8000/upload_catalog" \
  -F "file=@catalog.csv"
```

## Примітки

- Для роботи з зображеннями потрібно налаштувати LM Studio з LLaVA моделлю або використовувати GPT-4o
- Генерація зображень поки не реалізована (mock) - потрібна інтеграція з SDXL/FLUX
- ChromaDB зберігає документи локально в папці `chroma_db`
- Тули можна вмикати/вимикати через налаштування в frontend
