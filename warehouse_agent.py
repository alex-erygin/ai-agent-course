import openai  # Библиотека OpenAI для взаимодействия с API
import json    # Библиотека для работы с JSON (аргументы и ответы функций)
import os      # Библиотека для работы с операционной системой (доступ к переменным окружения)
from dotenv import load_dotenv  # Функция для загрузки переменных окружения из .env файла
from typing import List, Dict, Any, Callable  # Модуль для аннотаций типов (улучшает читаемость и проверку кода)

# --- Загрузка и Настройка ---

# Загрузка переменных окружения из файла .env (например, OPENAI_API_KEY)
load_dotenv()

# --- Константы ---

# Ключ API OpenAI, загруженный из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY.")

# ANSI escape-коды для цветного вывода в консоли
GREEN = '\033[92m'   # Зеленый (для ответов ассистента)
YELLOW = '\033[93m'  # Желтый (для ввода пользователя)
GRAY = '\033[90m'    # Серый (для сообщений о вызове функций)
RESET = '\033[0m'    # Сброс цвета

# Максимальное количество сообщений в истории диалога (не считая системного сообщения)
MAX_HISTORY = 10

# --- Инициализация Клиента OpenAI ---
# Создание клиента для взаимодействия с API OpenAI
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Хранилище Данных (In-memory) ---
# Словарь для хранения данных о товарах на складе (имя_товара: количество)
inventory: Dict[str, int] = {}

# --- Функции Управления Складом ---
# Эти функции будут вызываться моделью OpenAI для взаимодействия со складом

def add_item(item_name: str, quantity: int) -> str:
    """Добавляет товар и его количество на склад."""
    print(f"{GRAY}[Вызов функции] Добавление {quantity} шт. товара '{item_name}'{RESET}")
    inventory[item_name] = inventory.get(item_name, 0) + quantity
    # Возвращает JSON-строку с результатом операции
    return json.dumps({"status": "success", "message": f"Добавлено {quantity} шт. товара '{item_name}'."}, ensure_ascii=False)

def remove_item(item_name: str, quantity: int) -> str:
    """Удаляет товар со склада в указанном количестве."""
    print(f"{GRAY}[Вызов функции] Удаление {quantity} шт. товара '{item_name}'{RESET}")
    # Проверка наличия товара
    if item_name not in inventory:
        return json.dumps({"status": "error", "message": f"Товар '{item_name}' не найден."}, ensure_ascii=False)
    # Проверка достаточного количества
    if inventory[item_name] < quantity:
        return json.dumps({"status": "error", "message": f"Недостаточное количество товара '{item_name}'. В наличии: {inventory[item_name]}."}, ensure_ascii=False)

    inventory[item_name] -= quantity
    # Удаление товара из словаря, если количество стало нулевым
    if inventory[item_name] == 0:
        del inventory[item_name]
    # Возвращает JSON-строку с результатом операции
    return json.dumps({"status": "success", "message": f"Удалено {quantity} шт. товара '{item_name}'."}, ensure_ascii=False)

def get_inventory() -> str:
    """Возвращает отчет о текущем состоянии склада."""
    print(f"{GRAY}[Вызов функции] Получение отчета по складу{RESET}")
    if not inventory:
        return json.dumps({"status": "success", "inventory": "Склад пуст."}, ensure_ascii=False)
    # Возвращает JSON-строку с содержимым склада
    return json.dumps({"status": "success", "inventory": inventory}, ensure_ascii=False)

# --- Описание Инструментов (Функций) для OpenAI ---
# Структура данных, описывающая доступные модели функции, их параметры и назначение.
# Эта информация передается в API OpenAI.
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Добавить указанное количество товара на склад.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Название товара для добавления."},
                    "quantity": {"type": "integer", "description": "Количество единиц товара для добавления."},
                },
                "required": ["item_name", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Удалить указанное количество товара со склада.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Название товара для удаления."},
                    "quantity": {"type": "integer", "description": "Количество единиц товара для удаления."},
                },
                "required": ["item_name", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory",
            "description": "Получить отчет о текущем состоянии склада.",
            "parameters": {"type": "object", "properties": {}}, # Функция без параметров
        },
    },
]

# Словарь для сопоставления имен функций (строк) с реальными объектами функций Python.
# Используется для вызова нужной функции по имени, полученному от API.
available_functions: Dict[str, Callable[..., str]] = {
    "add_item": add_item,
    "remove_item": remove_item,
    "get_inventory": get_inventory,
}

# --- Логика Ведения Диалога ---

def _execute_tool_call(tool_call: Any, messages: List[Dict[str, Any]]) -> None:
    """
    Выполняет один вызов функции, запрошенный моделью.
    Определяет функцию, парсит аргументы, вызывает функцию и добавляет результат (или ошибку)
    обратно в историю сообщений (`messages`) с ролью 'tool'.
    """
    function_name = tool_call.function.name
    function_to_call = available_functions.get(function_name)
    # Подготовка сообщения с результатом вызова инструмента
    tool_message = {
        "tool_call_id": tool_call.id, # ID для связи с запросом модели
        "role": "tool",
        "name": function_name,
    }

    # Обработка случая, если функция не найдена
    if not function_to_call:
        tool_message["content"] = json.dumps(
            {"status": "error", "message": f"Функция '{function_name}' не найдена."}, ensure_ascii=False
        )
    else:
        # Попытка выполнить функцию
        try:
            # Аргументы от модели приходят в виде JSON-строки
            function_args = json.loads(tool_call.function.arguments)
            # Вызов реальной функции Python с аргументами
            function_response = function_to_call(**function_args)
            tool_message["content"] = function_response
        except json.JSONDecodeError:
            # Ошибка парсинга JSON-аргументов
            tool_message["content"] = json.dumps(
                {"status": "error", "message": "Неверный формат JSON аргументов."}, ensure_ascii=False
            )
        except TypeError as e:
            # Ошибка несоответствия аргументов функции
            tool_message["content"] = json.dumps(
                {"status": "error", "message": f"Неверные аргументы для функции {function_name}: {str(e)}"}, ensure_ascii=False
            )
        except Exception as e:
            # Любая другая ошибка при выполнении функции
            tool_message["content"] = json.dumps(
                {"status": "error", "message": f"Ошибка при выполнении функции {function_name}: {str(e)}"}, ensure_ascii=False
            )

    # Добавление результата вызова (или ошибки) в историю сообщений
    messages.append(tool_message)


def run_conversation(messages: List[Dict[str, Any]]) -> str | None:
    """
    Основная функция для ведения диалога с моделью OpenAI.
    Отправляет историю сообщений модели, обрабатывает потенциальные вызовы функций,
    выполняет их и возвращает итоговый ответ модели пользователю.
    """
    try:
        # Шаг 1: Отправка запроса модели с историей и доступными инструментами
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Модель OpenAI (можно сделать параметром)
            messages=messages,
            tools=tools,          # Передача описания функций
            tool_choice="auto",   # Модель сама решает, вызывать ли функцию
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls # Проверка, запросила ли модель вызов функции

        # Шаг 2: Если модель запросила вызов функции
        if tool_calls:
            # Добавляем ответ модели (с запросом на вызов) в историю
            messages.append(response_message)
            # Шаг 3: Выполнение каждой запрошенной функции
            for tool_call in tool_calls:
                _execute_tool_call(tool_call, messages)

            # Шаг 4: Отправка второго запроса модели с результатами вызова функций
            # Модель использует эти результаты для формулировки финального ответа.
            second_response = client.chat.completions.create(
                model="gpt-4o-mini", # Можно использовать ту же модель или другую
                messages=messages,
            )
            # Возвращаем текстовый ответ модели
            return second_response.choices[0].message.content
        else:
            # Если вызова функции не было, просто возвращаем текстовый ответ модели
            return response_message.content

    # Обработка различных ошибок API и других исключений
    except openai.APIError as e:
        print(f"{RED}Ошибка OpenAI API: {e}{RESET}")
        return f"Извините, произошла ошибка при связи с моделью ИИ: {e}"
    except openai.AuthenticationError:
        print(f"{RED}Ошибка аутентификации: Проверьте ваш OPENAI_API_KEY.{RESET}")
        return "Ошибка аутентификации: Убедитесь, что ваш OPENAI_API_KEY установлен правильно."
    except Exception as e:
        print(f"{RED}Произошла непредвиденная ошибка: {e}{RESET}")
        return "Извините, произошла непредвиденная ошибка."
    # Теоретически недостижимо при нормальной работе, но для полноты
    return None


# --- Основной Блок Выполнения ---

def main():
    """Главная функция запуска агента склада."""
    print("Агент Склада инициализирован.")
    print("Примеры команд: 'Добавь 5 яблок', 'Убери 2 банана', 'Покажи склад', 'Сколько яблок на складе?'")
    print("Введите 'выход' или 'exit' для завершения.")

    # Инициализация истории сообщений с системной инструкцией (промптом)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": (
        "Ты — полезный ассистент для управления складом. Ты говоришь по-русски. "
        "Используй доступные функции (add_item, remove_item, get_inventory) для управления инвентарем на основе запросов пользователя. "
        "Если пользователь просит добавить или убрать товары, вызови соответствующую функцию. "
        "Если пользователь спрашивает о текущих запасах или запрашивает отчет, используй функцию get_inventory. "
        "Подтверждай выполненные действия или предоставляй запрошенную информацию о складе. "
        "Если ты не можешь выполнить запрос с помощью функций, объясни почему. "
        "Если вызов функции возвращает статус ошибки, сообщи пользователю об ошибке."
    )}]

    # Основной цикл для взаимодействия с пользователем
    while True:
        try:
            # Получение ввода от пользователя
            user_input = input(f"{YELLOW}Вы: {RESET}")
        except EOFError: # Обработка Ctrl+D для выхода
             print("\nВыход.")
             break

        # Проверка команды выхода
        if user_input.lower() in ['выход', 'exit']:
            print("Завершение работы.")
            break

        # Добавление сообщения пользователя в историю
        messages.append({"role": "user", "content": user_input})

        # --- Управление размером истории ---
        # Оставляем системное сообщение и последние MAX_HISTORY сообщений (user/assistant/tool)
        # Это предотвращает слишком большой контекст для API
        if len(messages) > MAX_HISTORY + 1: # +1 для системного сообщения
             # Удаляем старые сообщения, сохраняя системное и последние MAX_HISTORY
             messages = [messages[0]] + messages[-(MAX_HISTORY):]


        # Получение ответа от ассистента (с возможным вызовом функций)
        assistant_response = run_conversation(messages)

        # Вывод ответа ассистента, если он получен
        if assistant_response:
            print(f"{GREEN}Ассистент: {assistant_response}{RESET}")
            # Добавление ответа ассистента в историю
            messages.append({"role": "assistant", "content": assistant_response})
        else:
            # Обработка случая, когда ответ не был получен (из-за ошибки)
            print(f"{RED}Ассистент: Не удалось получить ответ.{RESET}")
            # При необходимости можно удалить последнее сообщение пользователя,
            # чтобы избежать повторной отправки некорректного запроса.
            # messages.pop()


# Точка входа в программу: если скрипт запущен напрямую, вызываем main()
if __name__ == "__main__":
    main() 