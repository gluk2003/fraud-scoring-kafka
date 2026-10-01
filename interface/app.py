import json
import os
import uuid

import altair as alt
import pandas as pd
import psycopg2
import streamlit as st
from kafka import KafkaProducer

KAFKA_CONFIG = {
    'bootstrap_servers': os.getenv('KAFKA_BROKERS', 'kafka:9092'),
    'topic': os.getenv('KAFKA_TOPIC', 'transactions'),
}
PG_DSN = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'dbname': os.getenv('POSTGRES_DB', 'fraud'),
    'user': os.getenv('POSTGRES_USER', 'fraud'),
    'password': os.getenv('POSTGRES_PASSWORD', 'fraud'),
}

st.set_page_config(page_title='Fraud scoring', page_icon='🛡️', layout='wide')


def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f'Ошибка загрузки файла: {e}')
        return None


def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка транзакций в Kafka: одно сообщение = одна транзакция с уникальным ID"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            security_protocol='PLAINTEXT',
        )
        # to_dict('records') отдаёт нативные python-типы, NaN заменяем на null
        records = df.astype(object).where(df.notna(), None).to_dict('records')
        progress_bar = st.progress(0.0)
        total = len(records)
        for i, row in enumerate(records, start=1):
            producer.send(topic, value={'transaction_id': str(uuid.uuid4()), 'data': row})
            if i % 50 == 0 or i == total:
                progress_bar.progress(i / total)
        producer.flush()
        producer.close()
        return True
    except Exception as e:
        st.error(f'Ошибка отправки данных: {e}')
        return False


def query_db(sql):
    conn = psycopg2.connect(**PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    finally:
        conn.close()


def render_upload_tab():
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    uploaded_file = st.file_uploader('Загрузите CSV файл с транзакциями (формат test.csv)', type=['csv'])
    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files[uploaded_file.name] = {'status': 'Загружен', 'df': load_file(uploaded_file)}
        st.success(f'Файл {uploaded_file.name} успешно загружен!')

    if not st.session_state.uploaded_files:
        return

    st.subheader('🗂 Список загруженных файлов')
    for file_name, file_data in st.session_state.uploaded_files.items():
        cols = st.columns([4, 2, 2])
        with cols[0]:
            n_rows = len(file_data['df']) if file_data['df'] is not None else 0
            st.markdown(f'**Файл:** `{file_name}` ({n_rows} транзакций)')
            st.markdown(f"**Статус:** `{file_data['status']}`")
        with cols[2]:
            if st.button(f'Отправить {file_name}', key=f'send_{file_name}'):
                if file_data['df'] is None:
                    st.error('Файл не содержит данных')
                    continue
                with st.spinner('Отправка...'):
                    if send_to_kafka(file_data['df'], KAFKA_CONFIG['topic'], KAFKA_CONFIG['bootstrap_servers']):
                        st.session_state.uploaded_files[file_name]['status'] = 'Отправлен'
                        st.rerun()


def render_results_tab():
    st.caption('Данные читаются из витрины `scores` в PostgreSQL, которую заполняет сервис db_writer.')
    if not st.button('Посмотреть результаты', type='primary'):
        return

    try:
        frauds = query_db("""
            SELECT transaction_id, score, fraud_flag, created_at
            FROM scores
            WHERE fraud_flag = 1
            ORDER BY created_at DESC, id DESC
            LIMIT 10
        """)
        last = query_db("""
            SELECT score
            FROM scores
            ORDER BY created_at DESC, id DESC
            LIMIT 100
        """)
        total = query_db('SELECT count(*) AS n, coalesce(sum(fraud_flag), 0) AS n_fraud FROM scores')
    except Exception as e:
        st.error(f'Не удалось получить данные из PostgreSQL: {e}')
        return

    c1, c2 = st.columns(2)
    c1.metric('Всего транзакций в базе', int(total['n'][0]))
    c2.metric('Из них помечено как фрод', int(total['n_fraud'][0]))

    st.subheader('🚨 10 последних транзакций с fraud_flag = 1')
    if frauds.empty:
        st.info('Фродовых транзакций в базе пока нет.')
    else:
        st.dataframe(frauds, use_container_width=True, hide_index=True)

    st.subheader(f'📊 Распределение скоров последних {len(last)} транзакций')
    if last.empty:
        st.info('В базе пока нет транзакций — отправьте файл на вкладке «Отправка транзакций».')
    else:
        chart = alt.Chart(last).mark_bar().encode(
            x=alt.X('score:Q', bin=alt.Bin(extent=[0, 1], step=0.05), title='Скор модели'),
            y=alt.Y('count():Q', title='Количество транзакций'),
        )
        st.altair_chart(chart, use_container_width=True)


st.title('🛡️ Real-time fraud scoring')
tab_upload, tab_results = st.tabs(['📤 Отправка транзакций', '📈 Результаты'])
with tab_upload:
    render_upload_tab()
with tab_results:
    render_results_tab()
