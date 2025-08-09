# VK InsightGraph

Прототип для анализа коммуникаций в VK Teams.

## Установка

1. Установите Python 3.8+.
2. Установите зависимости: `pip install -r requirements.txt`.
3. Запустите приложение: `streamlit run app.py`.

## Использование

- Загрузите CSV/JSON с данными (пример: `mini_messages.csv`).
- Используйте фильтры в боковой панели для анализа.
- Переключайтесь между вкладками: Сообщения, Звонки, Встречи.

## Структура данных

CSV/JSON должен содержать: `sender`, `receiver`, `timestamp`, `channel` (опционально: `sender_department`, `receiver_department`, `call_duration`).
