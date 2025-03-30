#Требует java
import spacy
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from textstat import textstat
import logging
import re
from spellchecker import SpellChecker
import language_tool_python
from collections import defaultdict

# Настройки
API_TOKEN = '7935445281:AAFNhbIuOtDGhqZJlitA4T_sU5Ytx0tIIog'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация инструментов
try:
    nlp = spacy.load("en_core_web_sm")
    spell = SpellChecker()
    tool = language_tool_python.LanguageTool('en-US')
except Exception as e:
    logger.error(f"Ошибка инициализации: {e}")
    raise

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
user_messages = {}

# Детализированные уровни CEFR
CEFR_LEVELS = {
    "A1": {
        "name": "Начальный",
        "errors": "Очень частые ошибки всех типов",
        "grammar": "Только простейшие конструкции"
    },
    "A2": {
        "name": "Элементарный",
        "errors": "Много ошибок, но основная мысль понятна",
        "grammar": "Простые предложения, базовые времена"
    },
    "B1": {
        "name": "Средний",
        "errors": "Ошибки в сложных конструкциях",
        "grammar": "Основные времена, некоторые сложные предложения"
    },
    "B2": {
        "name": "Выше среднего",
        "errors": "Редкие ошибки, в основном в сложных случаях",
        "grammar": "Сложные конструкции, условные предложения"
    },
    "C1": {
        "name": "Продвинутый",
        "errors": "Минимальные ошибки, в основном стилистические",
        "grammar": "Все грамматические структуры уверенно"
    }
}


def analyze_errors(text):
    """Комплексный анализ ошибок с выделением неправильных слов"""
    # Орфографические ошибки
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    misspelled = list(spell.unknown(words))
    spelling_errors = {word: spell.correction(word) for word in misspelled}

    # Грамматические ошибки
    grammar_matches = tool.check(text)
    grammar_errors = [{
        'error': match.context,
        'message': match.message,
        'replacements': match.replacements[:3] if match.replacements else []
    } for match in grammar_matches]

    # Синтаксические ошибки (spaCy)
    doc = nlp(text)
    syntax_errors = []
    for sent in doc.sents:
        # Проверка порядка слов
        has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
        has_verb = any(tok.pos_ == "VERB" for tok in sent)

        if not has_subject:
            syntax_errors.append("Пропущено подлежащее")
        if not has_verb:
            syntax_errors.append("Пропущено сказуемое")

        # Проверка предлогов
        for tok in sent:
            if tok.dep_ in ("prep", "agent") and not any(t.head == tok for t in sent):
                syntax_errors.append(f"Проблема с предлогом: '{tok.text}'")

    return {
        "spelling_errors": spelling_errors,
        "grammar_errors": grammar_errors,
        "syntax_errors": syntax_errors,
        "total_errors": len(spelling_errors) + len(grammar_errors) + len(syntax_errors)
    }


def determine_level(error_analysis, text):
    """Определение уровня на основе анализа ошибок"""
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [token.text for token in doc if token.is_alpha]

    # Базовые метрики
    avg_sent_len = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
    unique_words = len(set(words)) / len(words) if words else 0

    # Оценка ошибок
    total_errors = error_analysis["total_errors"]
    error_rate = total_errors / len(words) if words else 0

    if error_rate > 0.3:
        return "A1"
    elif error_rate > 0.2:
        return "A2"
    elif error_rate > 0.1:
        return "B1"
    elif error_rate > 0.05:
        return "B2"
    else:
        return "C1"


@dp.message_handler(lambda message: message.text == "Оценить уровень")
async def evaluate_level(message: types.Message):
    user_id = message.from_user.id
    if not (messages := user_messages.get(user_id, [])):
        await message.reply("❌ Нет данных для анализа. Напишите несколько сообщений.")
        return

    full_text = " ".join(messages)
    if not re.search(r'[a-zA-Z]', full_text):
        await message.reply("🔠 Пожалуйста, напишите текст на английском.")
        return

    # Анализ ошибок
    error_analysis = analyze_errors(full_text)
    level = determine_level(error_analysis, full_text)
    level_info = CEFR_LEVELS.get(level, {})

    # Формирование отчета
    report_parts = []

    # 1. Орфографические ошибки
    if error_analysis["spelling_errors"]:
        spelling_report = ["\n✏️ Орфографические ошибки:"]
        for wrong, correct in error_analysis["spelling_errors"].items():
            spelling_report.append(f"{wrong} → {correct}")
        report_parts.append("\n".join(spelling_report))

    # 2. Грамматические ошибки
    if error_analysis["grammar_errors"]:
        grammar_report = ["\n🔠 Грамматические ошибки:"]
        for error in error_analysis["grammar_errors"][:5]:  # Ограничиваем количество примеров
            grammar_report.append(f"{error['error']} → {', '.join(error['replacements'][:3]) or 'нет предложений'}")
        report_parts.append("\n".join(grammar_report))

    # 3. Синтаксические ошибки
    if error_analysis["syntax_errors"]:
        syntax_report = ["\n📝 Синтаксические ошибки:"] + error_analysis["syntax_errors"][:5]
        report_parts.append("\n".join(syntax_report))

    # Итоговый отчет
    response = (
            f"📊 Уровень: {level} - {level_info.get('name', '')}\n"
            f"📌 Типичные ошибки: {level_info.get('errors', '')}\n"
            f"🔍 Всего ошибок: {error_analysis['total_errors']}\n\n"
            + "\n\n".join(report_parts) +
            "\n\n💡 Совет: " + get_advice(level) +
            "\n\nСообщения очищены. Можете начать новый анализ."
    )

    user_messages.pop(user_id, None)
    await message.reply(response)


def get_advice(level):
    advice = {
        "A1": "Учите базовые слова и простые предложения. Обратите внимание на правильное написание слов.",
        "A2": "Практикуйте Present/Past Simple. Составляйте больше предложений с базовой грамматикой.",
        "B1": "Изучайте Present Perfect и сложные предложения. Читайте адаптированную литературу.",
        "B2": "Работайте над сложными конструкциями. Практикуйте письменную речь с развернутыми предложениями.",
        "C1": "Совершенствуйте нюансы языка. Обращайте внимание на стилистику и идиомы."
    }
    return advice.get(level, "Регулярная практика - залог успеха!")


if __name__ == '__main__':
    logger.info("Бот запущен")
    executor.start_polling(dp, skip_updates=True)