import pandas as pd
import google.generativeai as genai
import logging
import pickle
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Путь для кэширования
CACHE_FILE = "llm_cache.pkl"
CACHE_DURATION = timedelta(hours=1)

def load_cache(analysis_type):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
            if cache.get('timestamp', datetime.min) + CACHE_DURATION > datetime.now():
                return cache.get(analysis_type, None)
    return None

def save_cache(response, analysis_type):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
    cache[analysis_type] = response
    cache['timestamp'] = datetime.now()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def generate_llm_prompt(group_summary, metrics, analysis_type):
    """
    Формируем компактный промпт для LLM
    """
    prompt = f"Ты — аналитик коммуникаций в VK Teams. Вот данные о группах общения и сотрудниках. Дай краткие рекомендации (до 100 слов) для улучшения командной динамики, используя простые термины.\n\n"

    # Информация о группах
    prompt += "Группы общения:\n"
    for group_id, row in group_summary.iterrows():
        prompt += f"- Группа {group_id}: {row['count']} чел., активность {row['degree']:.1f}, отдел: {row['main_department']}\n"

    # Информация о сотрудниках
    prompt += "\nКлючевые сотрудники:\n"
    for _, row in metrics.iterrows():
        if row['is_isolated'] or row['is_overloaded'] or row['is_bridge']:
            prompt += f"- {row['user']} ({row['department']}): {'изолирован' if row['is_isolated'] else ''}{'перегружен' if row['is_overloaded'] else ''}{'связующий' if row['is_bridge'] else ''}\n"

    if analysis_type == "messages":
        prompt += "\nПредложи, как вовлечь изолированных сотрудников, снизить нагрузку на перегруженных и укрепить связи между группами через чаты VK Teams."
    elif analysis_type == "calls":
        prompt += "\nПредложи, как оптимизировать звонки в VK Teams: маршрутизация звонков через связующих сотрудников и вовлечение изолированных."
    elif analysis_type == "meetings":
        prompt += "\nПредложи участников для встреч в VK WorkSpace, чтобы сбалансировать коммуникацию (например, добавить изолированных или связующих из разных групп)."

    return prompt

def call_llm(prompt, analysis_type):
    """
    Вызов Gemini API для анализа
    """
    # Проверяем кэш
    cached_response = load_cache(analysis_type)
    if cached_response:
        logger.info(f"Используется кэшированный ответ для {analysis_type}")
        return cached_response

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        logger.info(f"Ответ от Gemini API получен успешно для {analysis_type}")
        save_cache(response.text, analysis_type)
        return response.text
    except Exception as e:
        logger.error(f"Ошибка при вызове Gemini API: {str(e)}")
        # Резервная заглушка
        if analysis_type == "messages":
            return (
                "На основе анализа чатов VK Teams:\n"
                "- Для изолированных сотрудников (например, user7): назначьте их на проекты с активными группами (группа IT+Sales).\n"
                "- Для перегруженных (например, user1): делегируйте часть задач.\n"
                "- Для укрепления связей: организуйте чаты между Marketing+HR и IT+Sales."
            )
        elif analysis_type == "calls":
            return (
                "На основе анализа звонков VK Teams:\n"
                "- Маршрутизируйте звонки через связующих сотрудников (например, user1).\n"
                "- Для изолированных (например, user7): организуйте видеозвонки с группой IT+Sales.\n"
                "- Увеличьте частоту созвонов между Marketing+HR."
            )
        elif analysis_type == "meetings":
            return (
                "Подсказки для встреч в VK WorkSpace:\n"
                "- Добавьте user7 (изолирован) в встречи с группой IT+Sales.\n"
                "- Включите user1 (связующий) для связи Marketing+HR и IT+Sales.\n"
                "- Планируйте встречи для групп с низкой активностью (группа 2)."
            )

def analyze_with_llm(group_summary, metrics, analysis_type="messages"):
    """
    Анализируем данные с помощью LLM
    """
    prompt = generate_llm_prompt(group_summary, metrics, analysis_type)
    response = call_llm(prompt, analysis_type)
    return response