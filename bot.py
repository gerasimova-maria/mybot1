# bot.py
import os
import json
from urllib.parse import quote
import httpx
import sympy as sp
import re
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

# Загрузка переменных окружения из файла .env
load_dotenv()


# Конфигурация

TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN")      # токен Telegram‑бота
CHUTES_KEY = os.getenv("CHUTES_API_TOKEN")        # токен chutes.ai
CHUTES_URL = "https://llm.chutes.ai/v1/chat/completions"
HF_TOKEN   = os.getenv("HF_TOKEN")                # необязательный токен HuggingFace для распознавания изображений
HF_CAPTION_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

# Форматирование ответов: преобразование LaTeX и структурирование

def _latex_to_plain(text: str) -> str:
    """Грубое преобразование распространенного LaTeX в понятный plain-text.
    Не пытается идеально разобрать все конструкции, но сильно улучшает читаемость.
    """
    t = text
    # Простые замены символов и управляющих последовательностей
    replacements = [
        (r"\\pm", "±"),
        (r"\\times", "*"), (r"\\cdot", "*"),
        (r"\\leq", "≤"), (r"\\geq", "≥"), (r"\\neq", "≠"), (r"\\approx", "≈"),
        (r"\\infty", "∞"),
        (r"\\pi", "π"),
        (r"\\qquad", " "), (r"\\quad", " "), (r"\\,", " "), (r"\\;", " "), (r"\\:", " "),
    ]
    import re
    for pat, rep in replacements:
        t = re.sub(pat, rep, t)

    # Удаляем скобочные окружения и left/right
    t = re.sub(r"\\left|\\right", "", t)
    t = re.sub(r"\\\[|\\\]", "", t)
    t = re.sub(r"\\\(|\\\)", "", t)

    # \text{...} -> содержимое
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    # Тригонометрия и функции: \sin -> sin и т.п.
    t = re.sub(r"\\(sin|cos|tan|cot|sec|csc|log|ln)", r"\1", t)
    # Градусы: ^\circ -> ° и просто \circ -> °
    t = re.sub(r"\^\s*\\circ", "°", t)
    t = re.sub(r"\\circ", "°", t)

    # \frac{a}{b} -> (a)/(b)
    t = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", t)
    # \sqrt{...} -> √(...)
    t = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", t)
    # Степени: x^{2} -> x^2
    t = re.sub(r"([A-Za-z0-9\)])\^{\s*([^}]+)\s*}", r"\1^\2", t)

    # Удаляем большие массивы/окружения array, оставляя краткую пометку
    t = re.sub(r"\\begin\{array\}.*?\\end\{array\}", "[синтетическое деление опущено]", t, flags=re.S)
    # Убираем begin/end для прочих окружений
    t = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", t)

    # Маркдауны заголовков -> простые заголовки
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)
    # Жирный Markdown **...** убираем маркеры
    t = t.replace("**", "")
    # Разделители типа ---
    t = re.sub(r"^\s*-{3,}\s*$", "", t, flags=re.M)

    # Сохраняем переносы строк, убираем лишние пробелы внутри строк
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Удаляем двойные пустые строки до максимум двух подряд
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Нормализуем пробелы в строках
    t = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n"))
    t = t.strip()
    return t

def format_math_readable(answer: str) -> str:
    """Структурирует математический ответ:
    - конвертирует LaTeX в простой текст,
    - подчеркивает итог,
    - ограничивает шум и оставляет ключевые шаги.
    """
    import re
    plain = _latex_to_plain(answer)

    # Пытаемся выделить строку с итогом
    final_match = re.search(r"(?i)(ответ|итог)\s*:\s*(.+)", plain)
    final_line = final_match.group(2).strip() if final_match else None

    # Если нет явного итога — попробуем извлечь выражения вида x = ...
    if not final_line:
        eqs = re.findall(r"\b([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*([^\n]+)", plain)
        if eqs:
            # Берем до 3 выражений для краткости
            parts = [f"{v} = {val.strip()}" for v, val in eqs[:3]]
            final_line = "; ".join(parts)

    # Разделим на строки и оставим информативные
    lines = [ln.strip() for ln in re.split(r"\n|\. ", plain) if ln.strip()]
    key_lines = []
    for ln in lines:
        if len(key_lines) >= 6:
            break
        # отфильтруем шум и оставим шаги с вычислениями/факторизацией/формулой/решением
        if re.search(r"(корень|фактор|делени|формул|решени|равн|интеграл|производн|квадрат|итог|ответ|=|±|√|\^)", ln, flags=re.I):
            key_lines.append(ln)

    # Сборка компактного ответа
    out = []
    if final_line:
        out.append(f"Ответ: {final_line}")
    # Если ключевых строк нет, показываем урезанный оригинал
    if not key_lines:
        key_lines = lines[:4]
    if key_lines:
        out.append("Краткие шаги:")
        for ln in key_lines[:6]:
            out.append(f"- {ln}")

    # Если совсем коротко
    result = "\n".join(out).strip()
    return result or plain


# Клавиатура режимов

BUTTON_GENERAL = "Общие вопросы"
BUTTON_HELPER2 = "Математика"
BUTTON_HELPER3 = "Картинки"

def build_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BUTTON_GENERAL), KeyboardButton(BUTTON_HELPER2), KeyboardButton(BUTTON_HELPER3)]],
        resize_keyboard=True,
    )


# Функция обращения к chutes.ai

async def ask_chutes(user_msg: str) -> str:
    headers = {
        "Authorization": f"Bearer {CHUTES_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": user_msg}],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        chunks = []
        async with client.stream("POST", CHUTES_URL, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    choices = payload.get("choices", [])
                    if choices and len(choices) > 0:
                        text = choices[0].get("delta", {}).get("content", "")
                        if text:
                            chunks.append(text)
                except json.JSONDecodeError:
                    continue
                except (KeyError, IndexError) as e:
                    print(f"Ошибка при обработке ответа: {e}")
                    continue
    return "".join(chunks)

# Специализированный запрос к chutes.ai для математических текстовых задач
async def ask_chutes_math(user_msg: str) -> str:
    headers = {
        "Authorization": f"Bearer {CHUTES_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Ты математический ассистент. Решай задачи по шагам,"
                " выполняй вычисления точно, используй единицы измерения."
                " В конце выводи строку 'Ответ: <итог>' на русском."
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    body = {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        chunks = []
        async with client.stream("POST", CHUTES_URL, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    choices = payload.get("choices", [])
                    if choices and len(choices) > 0:
                        text = choices[0].get("delta", {}).get("content", "")
                        if text:
                            chunks.append(text)
                except json.JSONDecodeError:
                    continue
                except (KeyError, IndexError) as e:
                    print(f"Ошибка при обработке ответа: {e}")
                    continue
    return "".join(chunks)

# Математический помощник (SymPy, без внешних API)

def solve_math(user_msg: str) -> str:
    text = user_msg.strip().lower()
    # Нормализуем символ степени
    def normalize(expr: str) -> str:
        return expr.replace('^', '**')

    # Универсальная очистка текста от LaTeX/единиц и лишних символов
    def sanitize_text_for_math(s: str) -> str:
        s = s.replace('\\,', '')
        s = re.sub(r'\\text\{[^}]*\}', '', s)  # удаляем \text{...}
        s = s.replace('\\', '')
        s = s.replace('°', '')
        s = s.replace('π', 'pi')
        s = s.replace('√', 'sqrt')
        s = s.replace('–', '-')
        s = s.replace('—', '-')
        s = s.replace('−', '-')
        s = s.replace('×', '*').replace('·', '*')
        s = s.replace(',', '.')  # десятичные запятые
        # убрать единицы измерения после числа: "20 м/с", "9.8 м/с^2" → оставить число
        s = re.sub(r'(\d+(?:\.\d+)?)\s*[a-zA-Zа-яА-Я]+(?:\/[a-zA-Zа-яА-Я]+)?(?:\^?\d+)?', r'\1', s)
        # убрать содержимое в квадратных скобках, часто пояснения/единицы
        s = re.sub(r'\[[^\]]*\]', '', s)
        # удалить длинные последовательности кириллицы (текст), сохраняя короткие переменные
        s = re.sub(r'[а-яА-Я]{2,}', ' ', s)
        return s

    # Извлечь наиболее «математически похожую» подстроку
    def extract_math_candidate(s: str) -> str | None:
        s = sanitize_text_for_math(s)
        # приоритет: выражение после последнего двоеточия
        if ':' in s:
            tail = s.split(':')[-1].strip()
            if re.search(r'[0-9a-zA-Z\+\-\*/\^=]', tail):
                return tail
        # шаблон вида "уравнение: ..."
        m = re.search(r'уравнен[иея][^:]*:\s*([^\n\r]+)', s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # ищем фрагменты с допустимыми символами
        candidates = re.findall(r'[0-9a-zA-ZpiPI\+\-\*/\^\(\)\.,= ]+', s)
        # предпочитаем те, где есть хотя бы цифра или знак уравнения
        scored = [c.strip() for c in candidates if re.search(r'[0-9=]', c)]
        # отфильтруем простые присваивания вида "x = 1"
        scored = [c for c in scored if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*\d+(?:\.\d+)?', c)]
        if not scored:
            return None
        # берем самый длинный фрагмент
        return max(scored, key=len)

    # Извлечь числовые присваивания из текста, чтобы использовать их в вычислениях
    def extract_assignments_env(s: str) -> dict:
        s = sanitize_text_for_math(s)
        env = {}
        # ищем шаблоны вида "g = 9.8" или "a=2"; значение до следующего разделителя
        for var, val in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;:,\n\r]+)', s):
            try:
                env[var] = sp.sympify(normalize(val.strip()))
            except Exception:
                # игнорируем сложные/нечисловые присваивания
                pass
        return env

    try:
        env = extract_assignments_env(user_msg)
        # Решение уравнений: "реши x^2 - 4 = 0" (ключевое слово не обяз. в начале)
        if (text.startswith("реши ") or "реши" in text) or ("=" in text and any(k in text for k in ["реши", "уравнение"])):
            expr = normalize(user_msg.lower().split("реши", 1)[-1])
            if "=" in expr:
                left, right = expr.split("=", 1)
                left = left.split("реши", 1)[-1].strip()
                right = right.strip()
            else:
                # если нет правой части, решаем = 0
                left = expr.split("реши", 1)[-1].strip()
                right = "0"
            left_s = sp.sympify(left, locals=env)
            right_s = sp.sympify(right, locals=env)
            eq = sp.Eq(left_s, right_s)
            syms = list(eq.free_symbols)
            var = syms[0] if syms else sp.symbols('x')
            sol = sp.solve(eq, var)
            return f"Решения по {var}: {sol}"

        # Производная: "дифференцируй x^3" или "дифференцируй x^3 по x" (слово может быть не в начале)
        if (text.startswith("дифференцируй") or text.startswith("продифференцируй") or "дифференцируй" in text or "продифференцируй" in text):
            part = user_msg.lower().split("дифференцируй", 1)[-1]
            part = normalize(part)
            expr, var = part, 'x'
            if " по " in part:
                expr, var = part.split(" по ", 1)
                var = var.strip()
            f = sp.sympify(expr.strip(), locals=env)
            v = sp.symbols(var)
            d = sp.diff(f, v)
            return f"d/d{var} {sp.simplify(f)} = {sp.simplify(d)}"

        # Интеграл: "интеграл sin(x)" или "интеграл sin(x) по x" (слово может быть не в начале)
        if (text.startswith("интеграл") or text.startswith("проинтегрируй") or "интеграл" in text or "проинтегрируй" in text):
            part = user_msg.lower().split("интеграл", 1)[-1]
            part = normalize(part)
            expr, var = part, 'x'
            if " по " in part:
                expr, var = part.split(" по ", 1)
                var = var.strip()
            f = sp.sympify(expr.strip(), locals=env)
            v = sp.symbols(var)
            I = sp.integrate(f, v)
            return f"∫ {sp.simplify(f)} d{var} = {sp.simplify(I)} + C"

        # Вычисление выражения: "посчитай 2^10" (ключевое слово не обяз. в начале)
        if "посчитай" in text:
            part = user_msg.lower().split("посчитай", 1)[-1]
            expr = normalize(part)
            val = sp.simplify(sp.sympify(expr, locals=env))
            return f"Результат: {val}"

        # Общее: попытка упрощения/вычисления
        candidate = extract_math_candidate(user_msg)
        expr = normalize(candidate) if candidate else normalize(user_msg)
        val = sp.simplify(sp.sympify(expr, locals=env))
        return f"Результат: {val}"
    except Exception as e:
        return f"Не удалось обработать задачу: {str(e)}"


# Помощник картинок: генерация (Pollinations, без токена) и распознавание (HF, с токеном)

async def generate_image_url(prompt: str) -> str:
    # Бесплатная генерация через Pollinations: отдает URL картинки по тексту
    return f"https://image.pollinations.ai/prompt/{quote(prompt.strip())}"

async def caption_image(file_bytes: bytes) -> str:
    if not HF_TOKEN:
        return "Для распознавания изображений нужен бесплатный токен HuggingFace (HF_TOKEN в .env). Генерация по тексту доступна без токена."
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(HF_CAPTION_URL, headers=headers, content=file_bytes)
        if resp.status_code != 200:
            return f"Ошибка API распознавания: HTTP {resp.status_code}"
        data = resp.json()
        # Ответ обычно список с ключом generated_text
        try:
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            # иногда приходит dict с error
            if isinstance(data, dict) and "error" in data:
                return f"Ошибка API: {data['error']}"
            return "Не удалось распознать изображение."
        except Exception:
            return "Не удалось распознать изображение."

# Обработчик сообщений
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    # Обработка нажатий кнопок
    if user_text in (BUTTON_GENERAL, BUTTON_HELPER2, BUTTON_HELPER3):
        if user_text == BUTTON_GENERAL:
            context.user_data["model"] = "general"
            await update.message.reply_text("Режим 'Общие вопросы' активирован.")
        elif user_text == BUTTON_HELPER2:
            context.user_data["model"] = "math"
            await update.message.reply_text("Режим 'Математика' активирован. Напишите выражение или задачу.")
        else:  # BUTTON_HELPER3
            context.user_data["model"] = "images"
            await update.message.reply_text("Режим 'Картинки' активирован. Отправьте текст для генерации или фото для распознавания.")
        return

    await update.message.reply_text("Обрабатываю…")
    try:
        # Определяем активный режим
        active_model = context.user_data.get("model", "general")
        if active_model == "math":
            def looks_like_word_problem(t: str) -> bool:
                t = t.lower()
                keywords = [
                    "задача", "найти", "дано", "пусть", "скорость", "угол", "время", "расстояние",
                    "масса", "давление", "энергия", "вероятность", "процент", "площадь", "объем",
                    "радиус", "диаметр", "ускорение", "сила", "траектория", "координата", "координаты",
                    "вершины", "квадрат", "круг", "прямоугольник", "вектор", "ось", "геометрия",
                ]
                units = ["м/с", "м", "кг", "н", "дж", "рад", "°"]
                return any(k in t for k in keywords) or any(u in t for u in units)

            # Для текстовых задач LLM, для формул SymPy
            if looks_like_word_problem(user_text):
                answer = await ask_chutes_math(user_text)
            else:
                answer = solve_math(user_text)
                # Fallback: если парсинг формулы не удался, LLM-решение
                if isinstance(answer, str) and answer.startswith("Не удалось обработать задачу"):
                    answer = await ask_chutes_math(user_text)
            # Пост-обработка для читабельного вывода
            if isinstance(answer, str):
                answer = format_math_readable(answer)
        elif active_model == "images":
            # Генерация изображения по тексту
            img_url = await generate_image_url(user_text)
            await update.message.reply_photo(photo=img_url, caption="Сгенерировано по вашему промпту")
            return
        else:
            # Общие вопросы через chutes.ai
            answer = await ask_chutes(user_text)
        # Отправка длинных сообщений кусками, чтобы не упасть по лимиту Telegram
        async def reply_text_chunked(text: str):
            if not text:
                await update.message.reply_text("Извините, не удалось получить ответ от ИИ. Попробуйте еще раз.")
                return
            max_len = 3900
            for i in range(0, len(text), max_len):
                await update.message.reply_text(text[i:i+max_len])
        await reply_text_chunked(answer)
    except httpx.HTTPError as e:
        await update.message.reply_text("Ошибка подключения к серверу ИИ. Попробуйте позже.")
        print(f"Ошибка подключения: {e}")
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")
        print(f"Общая ошибка: {e}")
# Запуск
async def post_init(application: Application) -> None:
    """
    Безопасно отключает вебхук после инициализации приложения.
    """
    await application.bot.delete_webhook(drop_pending_updates=True)


if __name__ == "__main__":
    # Показ клавиатуры по команде /start
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        context.user_data["model"] = "general"
        await update.message.reply_text("Привет! Выберите режим:", reply_markup=build_main_keyboard())

    # Обработчик фотографий (для распознавания в режиме 'Картинки')
    async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        active_model = context.user_data.get("model", "general")
        if active_model != "images":
            await update.message.reply_text("Чтобы распознавать фото, включите режим 'Картинки'.")
            return
        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()
            file_bytes = await file.download_as_bytearray()
            caption = await caption_image(file_bytes)
            await update.message.reply_text(caption)
        except Exception as e:
            await update.message.reply_text(f"Не удалось обработать фото: {str(e)}")

    app = ApplicationBuilder().token(TG_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    # Запуск бота
    app.run_polling()