import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import networkx as nx
import json
import logging
import re
from analysis.clustering import add_clusters_to_metrics
from analysis.graph_builder import build_graph_from_messages
from analysis.graph_viz_plotly import plot_graph
from analysis.predictor import get_isolation_predictions
from analysis.llm_analyzer import analyze_with_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="VK InsightGraph", layout="wide")
st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"] {
        background-color: #424242 !important;
        color: #000000 !important;
    }
    .main {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .vk-logo {
        position: absolute;
        top: 10px;
        left: 10px;
        font-size: 2rem;
        color: #BBDEFB;
    }
    h1 {
        color: #BBDEFB !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        margin-left: 40px;
    }
    h2 {
        color: #90CAF9 !important;
        font-size: 1.5rem;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #616161;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stText, .stTextInput input, .stSelectbox, .stMultiselect, .stCheckbox label {
        color: #000000 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    [data-testid="stSidebar"] {
        background-color: #616161 !important;
        padding: 1.5rem;
    }
    .stButton button {
        background-color: #90CAF9 !important;
        color: #000000 !important;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
        border: 1px solid #BBDEFB !important;
    }
    .stButton button:hover {
        background-color: #BBDEFB !important;
        color: #000000 !important;
    }
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
        color: #000000 !important;
        background-color: #757575 !important;
    }
    .stDataFrame th {
        background-color: #616161 !important;
        color: #BBDEFB !important;
        font-weight: 600;
    }
    .stDataFrame tr:nth-child(even) {
        background-color: #424242 !important;
    }
    .stDataFrame td, .stDataFrame th {
        padding: 0.75rem;
        border: 1px solid #616161;
    }
    .recommendation-card {
        background-color: #616161 !important;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #90CAF9;
        color: #000000 !important;
    }
    .stPlotlyChart {
        background-color: #424242 !important;
        border-radius: 8px;
        padding: 1rem;
        min-height: 700px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .stTabs [role="tab"] {
        color: #B0BEC5 !important;
        font-weight: 500;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #BBDEFB !important;
        border-bottom: 2px solid #90CAF9 !important;
    }
    </style>
    <div class="vk-logo">☰</div>
""", unsafe_allow_html=True)

st.title("📊 VK InsightGraph — Анализ командной динамики")
st.markdown("""
Анализируйте коммуникации в VK Teams: выявляйте группы общения, изолированных сотрудников и перегруженных лидеров.

**Как использовать**:
1. Загрузите CSV/JSON с данными коммуникаций (пример: `mini_messages.csv`).
2. Используйте фильтры в боковой панели, чтобы выбрать отделы, группы или ключевых сотрудников.
3. Переключайтесь между вкладками: **Сообщения**, **Звонки**, **Встречи**.
4. Нажмите **Скачать граф как PNG** для сохранения визуализации.
""")

@st.cache_data
def load_data(uploaded_file, dashboard_path, parent_path):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, on_bad_lines='warn')
                logger.info("CSV loaded successfully")
                return df
            except Exception as e:
                st.error(f"Ошибка при загрузке CSV: {str(e)}")
                return None
        else:
            try:
                data = json.load(uploaded_file)
                df = pd.DataFrame(data)
                logger.info("JSON loaded successfully")
                return df
            except Exception as e:
                st.error(f"Ошибка при загрузке JSON: {str(e)}")
                return None
    else:
        if os.path.exists(dashboard_path):
            try:
                df = pd.read_csv(dashboard_path, on_bad_lines='warn')
                logger.info("Default CSV loaded from dashboard directory")
                return df
            except Exception as e:
                st.error(f"Ошибка при загрузке default CSV from dashboard: {str(e)}")
                return None
        elif os.path.exists(parent_path):
            try:
                df = pd.read_csv(parent_path, on_bad_lines='warn')
                logger.info("Default CSV loaded from parent directory")
                return df
            except Exception as e:
                st.error(f"Ошибка при загрузке default CSV from parent: {str(e)}")
                return None
        else:
            st.error("Не найдены файлы mini_messages.csv в dashboard или родительской директории. Загрузите данные вручную.")
            return None

with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV или JSON (sender, receiver, timestamp, channel)", type=["csv", "json"])

script_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_path = os.path.join(script_dir, "mini_messages.csv")
parent_path = os.path.join(script_dir, "..", "mini_messages.csv")
df = load_data(uploaded_file, dashboard_path, parent_path)

if df is None:
    st.stop()

required_columns = ['sender', 'receiver', 'timestamp']
if not all(col in df.columns for col in required_columns):
    st.error("Файл должен содержать колонки: sender, receiver, timestamp")
    st.stop()

if 'channel' not in df.columns:
    df['channel'] = 'message'
    logger.info("Added 'channel' column with default 'message'")

if 'call_duration' in df.columns:
    df['call_duration'] = df['call_duration'].where(df['channel'] == 'call', 0)
    logger.info("Filled missing 'call_duration' with 0 for non-call records")

@st.cache_data
def enrich_metrics_with_departments(df, metrics):
    departments = {}
    for _, row in df.iterrows():
        departments[row['sender']] = row.get('sender_department', 'Unknown')
        departments[row['receiver']] = row.get('receiver_department', 'Unknown')
    metrics['department'] = metrics['user'].map(departments).fillna('Unknown')
    return metrics

@st.cache_resource
def build_graph_cached(df, weight_col=None, export_metrics=False):
    return build_graph_from_messages(df, weight_col, export_metrics)

df_messages = df[df['channel'] == 'message'].copy()
df_calls = df[df['channel'] == 'call'].copy()

try:
    G_messages = build_graph_cached(df_messages)
    logger.info(f"G_messages created with {len(G_messages.nodes())} nodes and {len(G_messages.edges())} edges")
    metrics_messages = build_graph_cached(df_messages, export_metrics=True)
    logger.info(f"metrics_messages shape: {metrics_messages.shape}")
    metrics_messages = enrich_metrics_with_departments(df, metrics_messages)
    logger.info(f"metrics_messages departments: {metrics_messages['department'].unique().tolist()}")
    metrics_messages = add_clusters_to_metrics(G_messages, metrics_messages)
    logger.info(f"metrics_messages group_ids: {metrics_messages['group_id'].unique().tolist()}")
except Exception as e:
    st.error(f"Ошибка при обработке данных сообщений: {str(e)}")
    st.stop()

try:
    G_calls = build_graph_cached(df_calls, weight_col='call_duration')
    logger.info(f"G_calls created with {len(G_calls.nodes())} nodes and {len(G_calls.edges())} edges")
    metrics_calls = build_graph_cached(df_calls, weight_col='call_duration', export_metrics=True)
    metrics_calls = enrich_metrics_with_departments(df, metrics_calls)
    metrics_calls = add_clusters_to_metrics(G_calls, metrics_calls)
except Exception as e:
    st.error(f"Ошибка при обработке данных звонков: {str(e)}")
    metrics_calls = pd.DataFrame()

with st.sidebar:
    st.header("🔍 Фильтры")
    if 'group_filter' not in st.session_state:
        st.session_state.group_filter = metrics_messages['group_id'].unique().tolist()
    if 'department_filter' not in st.session_state:
        st.session_state.department_filter = metrics_messages['department'].unique().tolist()
    if 'show_isolated' not in st.session_state:
        st.session_state.show_isolated = False
    if 'show_overloaded' not in st.session_state:
        st.session_state.show_overloaded = False
    if 'show_bridges' not in st.session_state:
        st.session_state.show_bridges = False

    group_filter = st.multiselect("Группы общения", sorted(metrics_messages['group_id'].unique()), default=st.session_state.group_filter, key="group_filter")
    department_filter = st.multiselect("Отделы", sorted(metrics_messages['department'].unique()), default=st.session_state.department_filter, key="department_filter")
    show_isolated = st.checkbox("Показать только изолированных", value=st.session_state.show_isolated, key="show_isolated")
    show_overloaded = st.checkbox("Показать только перегруженных", value=st.session_state.show_overloaded, key="show_overloaded")
    show_bridges = st.checkbox("Показать только связующих", value=st.session_state.show_bridges, key="show_bridges")
    
    if st.button("Сбросить фильтры"):
        st.session_state.group_filter = metrics_messages['group_id'].unique().tolist()
        st.session_state.department_filter = metrics_messages['department'].unique().tolist()
        st.session_state.show_isolated = False
        st.session_state.show_overloaded = False
        st.session_state.show_bridges = False
        st.rerun()

filtered_metrics = metrics_messages.copy()
if group_filter:
    filtered_metrics = filtered_metrics[filtered_metrics['group_id'].isin(group_filter)]
if department_filter:
    filtered_metrics = filtered_metrics[filtered_metrics['department'].isin(department_filter)]
if show_isolated:
    filtered_metrics = filtered_metrics[filtered_metrics['is_isolated']]
if show_overloaded:
    filtered_metrics = filtered_metrics[filtered_metrics['is_overloaded']]
if show_bridges:
    filtered_metrics = filtered_metrics[filtered_metrics['is_bridge']]
logger.info(f"Filtered metrics users: {filtered_metrics['user'].tolist()}")

tab1, tab2, tab3 = st.tabs(["📩 Сообщения", "📞 Звонки", "🗓️ Встречи"])

with tab1:
    st.subheader("📩 Анализ сообщений")
    st.markdown("Узнайте, как сотрудники общаются в чатах VK Teams.")
    try:
        group_summary = metrics_messages.groupby('group_id').agg({
            'user': 'count',
            'degree': 'mean',
            'clustering': 'mean',
            'department': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).rename(columns={'user': 'count', 'department': 'main_department'})
        st.dataframe(group_summary.rename(columns={
            'count': 'Количество сотрудников',
            'degree': 'Средняя активность',
            'clustering': 'Плотность связей',
            'main_department': 'Основной отдел'
        }), use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка при создании group_summary: {str(e)}")

    st.markdown("**Рекомендации по группам**:")
    for group_id in group_summary.index:
        count = group_summary.loc[group_id, 'count']
        mean_degree = group_summary.loc[group_id, 'degree']
        main_dept = group_summary.loc[group_id, 'main_department']
        if count <= 2:
            st.markdown(f"<div class='recommendation-card'>Группа {group_id} ({main_dept}, {count} чел.): Мало участников, добавьте сотрудников.</div>", unsafe_allow_html=True)
        elif mean_degree < metrics_messages['degree'].mean():
            st.markdown(f"<div class='recommendation-card'>Группа {group_id} ({main_dept}, {count} чел.): Низкая активность, инициируйте обсуждения.</div>", unsafe_allow_html=True)

    st.markdown("**Ключевые связующие сотрудники**:")
    bridges = metrics_messages[metrics_messages['is_bridge']]
    for user in bridges['user']:
        st.markdown(f"<div class='recommendation-card'>{user}: Связующий сотрудник, поддерживайте активность.</div>", unsafe_allow_html=True)

    st.subheader("🧠 Инсайты от ИИ (сообщения)")
    try:
        llm_recommendations = analyze_with_llm(group_summary, metrics_messages, analysis_type="messages")
        st.markdown(f"<div class='recommendation-card'>{llm_recommendations}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Ошибка в анализе ИИ: {str(e)}")

    st.subheader("🕸️ Граф коммуникаций")
    if filtered_metrics.empty:
        st.warning("Нет данных для построения графа после фильтрации.")
    else:
        users_to_plot = filtered_metrics['user'].tolist()
        department_colors = {
            'IT': '#64B5F6',
            'Finance': '#FFCA28',
            'Sales': '#4CAF50',
            'HR': '#EF5350',
            'Marketing': '#AB47BC',
            'Unknown': '#B0BEC5'
        }
        if not users_to_plot:
            st.warning("Нет пользователей для отображения графа после фильтрации.")
        elif not G_messages.nodes():
            st.warning("Граф пуст: нет связей между пользователями.")
        else:
            try:
                logger.info(f"Plotting graph with {len(users_to_plot)} users")
                fig = plot_graph(G_messages, filtered_metrics, users_to_plot, department_colors)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True, key=f"graph_{hash(tuple(users_to_plot))}")
                    if st.button("Скачать граф как PNG"):
                        fig.write_image("graph.png")
                        with open("graph.png", "rb") as file:
                            st.download_button("Скачать PNG", file, "graph.png")
                else:
                    st.warning("Нет связей между выбранными пользователями. Попробуйте изменить фильтры.")
            except Exception as e:
                st.error(f"Ошибка при построении графа: {str(e)}")
                logger.error(f"Graph plotting error: {str(e)}", exc_info=True)

with tab2:
    st.subheader("🔍 Анализ звонков")
    st.markdown("Узнайте, как сотрудники взаимодействуют через видеозвонки VK Teams.")
    if metrics_calls.empty:
        st.warning("Нет данных о звонках для анализа.")
    else:
        try:
            group_summary_calls = metrics_calls.groupby('group_id').agg({
                'user': 'count',
                'degree': 'mean',
                'clustering': 'mean',
                'department': lambda x: x.mode()[0] if not x.empty else 'Unknown'
            }).rename(columns={'user': 'count', 'department': 'main_department'})
            st.dataframe(group_summary_calls.rename(columns={
                'count': 'Количество сотрудников',
                'degree': 'Средняя активность (звонки)',
                'clustering': 'Плотность связей',
                'main_department': 'Основной отдел'
            }), use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка при создании group_summary_calls: {str(e)}")

        st.markdown("**Рекомендации по звонкам**:")
        if not group_summary_calls.empty:
            for group_id in group_summary_calls.index:
                count = group_summary_calls.loc[group_id, 'count']
                mean_degree = group_summary_calls.loc[group_id, 'degree']
                main_dept = group_summary_calls.loc[group_id, 'main_department']
                if count <= 2:
                    st.markdown(f"<div class='recommendation-card'>Группа {group_id} ({main_dept}, {count} чел.): Мало звонков, организуйте видеовстречи.</div>", unsafe_allow_html=True)
                elif mean_degree < metrics_calls['degree'].mean():
                    st.markdown(f"<div class='recommendation-card'>Группа {group_id} ({main_dept}, {count} чел.): Низкая активность звонков, предложите регулярные созвоны.</div>", unsafe_allow_html=True)

        st.markdown("**Маршрутизация звонков**:")
        bridges = metrics_calls[metrics_calls['is_bridge']]
        for user in bridges['user']:
            st.markdown(f"<div class='recommendation-card'>Переключите звонки на {user}: связующий сотрудник для разных команд.</div>", unsafe_allow_html=True)

        st.subheader("🧠 Инсайты от ИИ (звонки)")
        try:
            llm_recommendations = analyze_with_llm(group_summary_calls, metrics_calls, analysis_type="calls")
            st.markdown(f"<div class='recommendation-card'>{llm_recommendations}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ошибка в анализе ИИ: {str(e)}")

with tab3:
    st.subheader("🗓️ Подсказки для встреч")
    st.markdown("ИИ поможет выбрать участников для встреч в календаре VK WorkSpace.")
    try:
        llm_meeting_suggestions = analyze_with_llm(group_summary, metrics_messages, analysis_type="meetings")
        st.markdown(f"<div class='recommendation-card'>{llm_meeting_suggestions}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Ошибка в анализе ИИ: {str(e)}")

st.subheader("📝 Рекомендации")
if filtered_metrics.empty:
    st.warning("Нет данных для отображения после фильтрации.")
else:
    isolated = filtered_metrics[filtered_metrics['is_isolated']]
    overloaded = filtered_metrics[filtered_metrics['is_overloaded']]
    bridges = filtered_metrics[filtered_metrics['is_bridge']]

    if not isolated.empty:
        st.markdown("**Изолированные сотрудники**:")
        for user in isolated['user']:
            st.markdown(f"<div class='recommendation-card'>{user}: Увеличьте вовлечённость через встречи.</div>", unsafe_allow_html=True)
    if not overloaded.empty:
        st.markdown("**Перегруженные сотрудники**:")
        for user in overloaded['user']:
            st.markdown(f"<div class='recommendation-card'>{user}: Делегируйте задачи или снизите нагрузку.</div>", unsafe_allow_html=True)
    if not bridges.empty:
        st.markdown("**Ключевые связующие сотрудники**:")
        for user in bridges['user']:
            st.markdown(f"<div class='recommendation-card'>{user}: Поддерживайте активность, связывает команды.</div>", unsafe_allow_html=True)

st.subheader("🔮 Предсказания изоляции")
try:
    high_risk = get_isolation_predictions(metrics_messages)
    if not high_risk.empty:
        st.markdown("**Сотрудники с риском изоляции**:")
        for _, row in high_risk.iterrows():
            st.markdown(f"<div class='recommendation-card'>{row['user']}: Риск изоляции {row['isolation_risk']:.2%}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='recommendation-card'>Нет сотрудников с высоким риском изоляции.</div>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Ошибка в предсказаниях: {str(e)}")

st.subheader("📈 Метрики участников")
st.dataframe(
    filtered_metrics.sort_values(by="degree", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=500
)

with st.sidebar:
    st.header("🔎 Поиск участника")
    search_name = st.text_input("Введите имя (user_id)")
    if search_name:
        escaped_search = re.escape(search_name)
        result = metrics_messages[metrics_messages['user'].str.contains(escaped_search, case=False, na=False)]
        st.markdown(f"🔍 Найдено: {len(result)}")
        st.dataframe(result)
