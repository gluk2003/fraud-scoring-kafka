"""Чтение транзакций из Kafka и запись результатов скоринга в Kafka."""
import json
import logging
import time

from confluent_kafka import Consumer, KafkaException, Producer

logger = logging.getLogger(__name__)


def wait_for_topics(bootstrap_servers: str, topics: list, timeout: int = 120) -> None:
    """Ждём, пока брокер поднимется и kafka-setup создаст нужные топики."""
    probe = Consumer({'bootstrap.servers': bootstrap_servers, 'group.id': 'topic-probe'})
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                existing = probe.list_topics(timeout=5).topics
                if all(t in existing for t in topics):
                    logger.info('Kafka topics are ready: %s', topics)
                    return
            except KafkaException as e:
                logger.info('Kafka is not available yet: %s', e)
            logger.info('Waiting for topics %s ...', topics)
            time.sleep(3)
    finally:
        probe.close()
    raise RuntimeError(f'Topics {topics} did not appear in {timeout}s')


class TransactionConsumer:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        })
        self.consumer.subscribe([topic])
        logger.info('Subscribed to topic "%s" as group "%s"', topic, group_id)

    def poll_batch(self, max_messages: int = 500, timeout: float = 1.0) -> list:
        """Возвращает список распарсенных сообщений {transaction_id, data}."""
        messages = self.consumer.consume(num_messages=max_messages, timeout=timeout)
        batch = []
        for msg in messages:
            if msg.error():
                logger.error('Kafka error: %s', msg.error())
                continue
            try:
                payload = json.loads(msg.value().decode('utf-8'))
                if 'transaction_id' not in payload or 'data' not in payload:
                    raise ValueError('message must contain "transaction_id" and "data"')
                batch.append(payload)
            except Exception as e:
                logger.error('Skipping malformed message at offset %s: %s', msg.offset(), e)
        return batch

    def close(self) -> None:
        self.consumer.close()


class ScoreProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self.producer = Producer({'bootstrap.servers': bootstrap_servers})

    def send(self, records: list) -> None:
        """records: [{transaction_id, score, fraud_flag}, ...]"""
        for rec in records:
            self.producer.produce(
                self.topic,
                key=rec['transaction_id'],
                value=json.dumps(rec).encode('utf-8'),
            )
            self.producer.poll(0)
        self.producer.flush()
