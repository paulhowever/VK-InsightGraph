import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_isolation_predictor(metrics_df):
    """
    Обучаем модель для предсказания изолированных сотрудников
    """
    # Debug: Log available columns
    logger.info(f"Metrics columns: {metrics_df.columns.tolist()}")
    
    # Подготовка данных
    features = ['degree', 'betweenness', 'eigenvector', 'clustering']
    X = metrics_df[features]
    y = metrics_df['is_isolated']

    # Проверка на наличие нескольких классов
    if len(y.unique()) < 2:
        # Если только один класс, возвращаем нулевые вероятности изоляции
        metrics_df['isolation_risk'] = np.zeros(len(metrics_df))
        return metrics_df, None, 0.0

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка точности
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.2f}")

    # Предсказания для всех сотрудников
    probas = model.predict_proba(X)
    # Если модель возвращает вероятности для двух классов, берём вероятность изоляции
    metrics_df['isolation_risk'] = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
    return metrics_df, model, accuracy

def get_isolation_predictions(metrics_df):
    """
    Возвращаем сотрудников с высоким риском изоляции
    """
    metrics_df, _, _ = train_isolation_predictor(metrics_df)
    high_risk = metrics_df[metrics_df['isolation_risk'] > 0.7][['user', 'isolation_risk']]
    return high_risk