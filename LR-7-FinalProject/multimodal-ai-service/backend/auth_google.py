"""
Скрипт для авторизації Google API
Запустіть цей скрипт один раз для отримання token.json
"""

import os
from google_auth_oauthlib.flow import InstalledAppFlow

CREDENTIALS_PATH = './credentials.json'
TOKEN_PATH = './token.json'

SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.send'
]

def force_login():
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"❌ ПОМИЛКА: Файл не знайдено за адресою: {CREDENTIALS_PATH}")
        print("Перевір шлях або завантаж файл credentials.json у папку backend/")
        return

    print("🚀 Починаємо авторизацію...")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        # Запустити локальний сервер для авторизації
        creds = flow.run_local_server(port=0)
        
        # Зберігаємо токен
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
            
        print("\n✅ УСПІХ! Авторизацію пройдено. Файл token.json створено.")
        print(f"   Токен збережено в: {TOKEN_PATH}")
        
    except Exception as e:
        print(f"\n❌ Помилка під час входу: {e}")
        print("\nАльтернативний спосіб:")
        print("1. Відкрийте браузер")
        print("2. Перейдіть за посиланням, яке з'явиться")
        print("3. Авторизуйтеся та скопіюйте код")
        print("4. Вставте код в консоль")

if __name__ == "__main__":
    force_login()

