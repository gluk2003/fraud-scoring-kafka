"""Сервис выгрузки результатов скоринга: Kafka (топик scores) -> PostgreSQL."""
import json
import logging
import os
import time

import psycopg2
from confluent_kafka import Consumer, KafkaException
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_writer')

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
SCORES_TOPIC = os.getenv('KAFKA_SCORES_TOPIC', 'scores')
PG_DSN = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'dbname': os.getenv('POSTGRES_DB', 'fraud'),
    'user': os.getenv('POSTGRES_USER', 'fraud'),
    'password': os.getenv('POSTGRES_PASSWORD', 'fraud'),
}

INSERT_SQL = """
    INSERT INTO scores (transaction_id, score, fraud_flag)
    VALUES %s
    ON CONFLICT (transaction_id) DO UPDATE
        SET score = EXCLUDED.score, fraud_flag = EXCLUDED.fraud_flag
"""


def connect_db(retries: int = 30):
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(**PG_DSN)
            logger.info('Connected to PostgreSQL %s:%s/%s', PG_DSN['host'], PG_DSN['port'], PG_DSN['dbname'])
            return conn
        except psycopg2.OperationalError as e:
            logger.info('PostgreSQL is not ready (attempt %d/%d): %s', attempt, retries, e)
            time.sleep(2)
    raise RuntimeError('Could not connect to PostgreSQL')


def wait_for_topic(timeout: int = 120) -> None:
    probe = Consumer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS, 'group.id': 'topic-probe'})
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                if SCORES_TOPIC in probe.list_topics(timeout=5).topics:
                    return
            except KafkaException as e:
                logger.info('Kafka is not available yet: %s', e)
            logger.info('Waiting for topic "%s" ...', SCORES_TOPIC)
            time.sleep(3)
    finally:
        probe.close()
    raise RuntimeError(f'Topic {SCORES_TOPIC} did not appear in {timeout}s')


def parse(msg):
    rec = json.loads(msg.value().decode('utf-8'))
    return str(rec['transaction_id']), float(rec['score']), int(rec['fraud_flag'])


def write_rows(conn, rows):
    """Пишет батч в БД; при ошибке переподключается и повторяет тот же батч."""
    while True:
        try:
            with conn.cursor() as cur:
                execute_values(cur, INSERT_SQL, rows)
            conn.commit()
            return conn
        except psycopg2.Error:
            logger.exception('Failed to write batch, reconnecting')
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(2)
            conn = connect_db()


def main():
    conn = connect_db()
    wait_for_topic()
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'db-writer',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,  # коммитим offset только после записи в БД
    })
    consumer.subscribe([SCORES_TOPIC])
    logger.info('Listening topic "%s"', SCORES_TOPIC)

    try:
        while True:
            messages = consumer.consume(num_messages=500, timeout=1.0)
            if not messages:
                continue
            rows = []
            for msg in messages:
                if msg.error():
                    logger.error('Kafka error: %s', msg.error())
                    continue
                try:
                    rows.append(parse(msg))
                except Exception as e:
                    logger.error('Skipping malformed message at offset %s: %s', msg.offset(), e)
            if rows:
                conn = write_rows(conn, rows)
                logger.info('Saved %d scores to PostgreSQL', len(rows))
            consumer.commit(asynchronous=False)
    except KeyboardInterrupt:
        logger.info('Stopped by user')
    finally:
        consumer.close()
        conn.close()


if __name__ == '__main__':
    main()
