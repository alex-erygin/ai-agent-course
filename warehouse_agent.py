import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI()

# In-memory storage for the warehouse inventory
inventory = {}

# ANSI escape codes for colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
GRAY = '\033[90m' # For function calls (looks silver/gray)
RESET = '\033[0m'

def add_item(item_name: str, quantity: int):
    """Adds an item to the warehouse inventory."""
    print(f"{GRAY}[Вызов функции] Добавление {quantity} шт. товара '{item_name}'{RESET}")
    # Simple inventory update logic
    inventory[item_name] = inventory.get(item_name, 0) + quantity
    return json.dumps({"status": "success", "message": f"Добавлено {quantity} шт. товара '{item_name}'."}, ensure_ascii=False) # ensure_ascii=False for Russian characters

def remove_item(item_name: str, quantity: int):
    """Removes an item from the warehouse inventory."""
    print(f"{GRAY}[Вызов функции] Удаление {quantity} шт. товара '{item_name}'{RESET}")
    # Simple inventory update logic with error handling
    if item_name not in inventory:
        return json.dumps({"status": "error", "message": f"Товар '{item_name}' не найден."}, ensure_ascii=False)
    if inventory[item_name] < quantity:
        return json.dumps({"status": "error", "message": f"Недостаточное количество товара '{item_name}'. В наличии: {inventory[item_name]}."}, ensure_ascii=False)
    inventory[item_name] -= quantity
    if inventory[item_name] == 0:
        del inventory[item_name] # Remove item if quantity is zero
    return json.dumps({"status": "success", "message": f"Удалено {quantity} шт. товара '{item_name}'."}, ensure_ascii=False)


def get_inventory():
    """Gets the current warehouse inventory report."""
    print(f"{GRAY}[Вызов функции] Получение отчета по складу{RESET}")
    # Simple inventory reporting logic
    if not inventory:
        return json.dumps({"status": "success", "inventory": "Склад пуст."}, ensure_ascii=False)
    # Ensure keys and values with Russian text are handled correctly if needed in the future
    return json.dumps({"status": "success", "inventory": inventory}, ensure_ascii=False)

# Define the functions available to the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Добавить указанное количество товара на склад.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Название товара для добавления.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Количество единиц товара для добавления.",
                    },
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
                    "item_name": {
                        "type": "string",
                        "description": "Название товара для удаления.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Количество единиц товара для удаления.",
                    },
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
            "parameters": {"type": "object", "properties": {}}, # No parameters needed
        },
    },
]

available_functions = {
    "add_item": add_item,
    "remove_item": remove_item,
    "get_inventory": get_inventory,
}

def run_conversation(messages):
    """Runs the conversation with the OpenAI model, handling function calls."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125", # Or your preferred model
            messages=messages,
            tools=tools,
            tool_choice="auto", # auto is default, but we'll be explicit
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            messages.append(response_message) # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                if not function_to_call:
                     messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"status": "error", "message": f"Функция '{function_name}' не найдена."}, ensure_ascii=False),
                        }
                    )
                     continue

                try:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    ) # extend conversation with function response
                except json.JSONDecodeError:
                     messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"status": "error", "message": "Неверный формат JSON аргументов."}, ensure_ascii=False),
                        }
                    )
                except TypeError as e: # Handle cases where arguments don't match function signature
                     messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"status": "error", "message": f"Неверные аргументы для функции {function_name}: {str(e)}"}, ensure_ascii=False),
                        }
                    )
                except Exception as e: # Catch other potential errors during function execution
                     messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps({"status": "error", "message": f"Ошибка при выполнении функции {function_name}: {str(e)}"}, ensure_ascii=False),
                        }
                    )

            # get a new response from the model where it can see the function response
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages,
            )
            return second_response.choices[0].message.content
        else:
            # If no function call, return the model's response directly
            return response_message.content

    except openai.APIError as e:
        print(f"Ошибка OpenAI API: {e}")
        return f"Извините, произошла ошибка при связи с моделью ИИ: {e}"
    except openai.AuthenticationError:
        print("Ошибка аутентификации: Проверьте ваш OPENAI_API_KEY.")
        return "Ошибка аутентификации: Убедитесь, что ваш OPENAI_API_KEY установлен правильно."
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return "Извините, произошла непредвиденная ошибка."


if __name__ == "__main__":
    print("Агент Склада инициализирован.")
    print("Примеры команд: 'Добавь 5 яблок', 'Убери 2 банана', 'Покажи склад', 'Сколько яблок на складе?'")
    print("Введите 'выход' или 'exit' для завершения.")
    # Initial system message to guide the assistant in Russian
    messages = [{"role": "system", "content": (
        "Ты — полезный ассистент для управления складом. Ты говоришь по-русски. "
        "Используй доступные функции (add_item, remove_item, get_inventory) для управления инвентарем на основе запросов пользователя. "
        "Если пользователь просит добавить или убрать товары, вызови соответствующую функцию. "
        "Если пользователь спрашивает о текущих запасах или запрашивает отчет, используй функцию get_inventory. "
        "Подтверждай выполненные действия или предоставляй запрошенную информацию о складе. "
        "Если ты не можешь выполнить запрос с помощью функций, объясни почему. "
        "Если вызов функции возвращает статус ошибки, сообщи пользователю об ошибке."
        )}]
    while True:
        # Use yellow for user input prompt
        user_input = input(f"{YELLOW}Вы: {RESET}")
        # Use 'выход' or 'exit' to quit, case-insensitive
        if user_input.lower() in ['выход', 'exit']:
            break
        messages.append({"role": "user", "content": user_input})
        # Remove the last assistant message if it exists to avoid duplication before calling run_conversation
        if len(messages) > 1 and messages[-1]["role"] == "assistant":
             messages.pop()

        assistant_response = run_conversation(messages)
        # Print assistant response only if it's not None or empty, and in green
        if assistant_response:
            print(f"{GREEN}Ассистент: {assistant_response}{RESET}")
            # Add assistant's response to history *after* processing and printing
            messages.append({"role": "assistant", "content": assistant_response})

        # Optional: Limit message history size
        MAX_HISTORY = 10
        if len(messages) > MAX_HISTORY:
            messages = messages[0:1] + messages[-MAX_HISTORY+1:] 