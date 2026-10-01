"""Точка входа сервиса скоринга: Kafka -> препроцессинг -> модель -> Kafka."""
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from kafka_io import ScoreProducer, TransactionConsumer, wait_for_topics  # noqa: E402
from preprocessing import load_artifacts, run_preproc  # noqa: E402
from scorer import Scorer  # noqa: E402

LOG_DIR = os.getenv('LOG_DIR', '/app/logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, 'service.log')), logging.StreamHandler()],
)
logger = logging.getLogger('fraud_detector')

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TRANSACTIONS_TOPIC = os.getenv('KAFKA_TRANSACTIONS_TOPIC', 'transactions')
SCORES_TOPIC = os.getenv('KAFKA_SCORES_TOPIC', 'scores')
MODEL_DIR = os.getenv('MODEL_DIR', '/app/models')


class ProcessingService:
    def __init__(self):
        self.artifacts = load_artifacts(os.path.join(MODEL_DIR, 'preprocessing_artifacts.json'))
        self.scorer = Scorer(os.path.join(MODEL_DIR, 'model.cbm'), os.path.join(MODEL_DIR, 'model_meta.json'))
        wait_for_topics(KAFKA_BOOTSTRAP_SERVERS, [TRANSACTIONS_TOPIC, SCORES_TOPIC])
        self.consumer = TransactionConsumer(KAFKA_BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC, 'ml-scorer')
        self.producer = ScoreProducer(KAFKA_BOOTSTRAP_SERVERS, SCORES_TOPIC)

    def score_batch(self, batch: list) -> list:
        ids = [m['transaction_id'] for m in batch]
        raw = pd.DataFrame([m['data'] for m in batch])
        features = run_preproc(raw, self.artifacts)
        preds = self.scorer.make_pred(features)
        return [
            {'transaction_id': tid, 'score': float(s), 'fraud_flag': int(f)}
            for tid, s, f in zip(ids, preds['score'], preds['fraud_flag'])
        ]

    def run(self):
        logger.info('Scoring loop started: %s -> %s', TRANSACTIONS_TOPIC, SCORES_TOPIC)
        while True:
            batch = self.consumer.poll_batch()
            if not batch:
                continue
            try:
                results = self.score_batch(batch)
            except Exception:
                # Если батч целиком не обработался, скорим по одному, чтобы
                # одна битая транзакция не блокировала остальные
                logger.exception('Batch scoring failed, falling back to per-message scoring')
                results = []
                for msg in batch:
                    try:
                        results.extend(self.score_batch([msg]))
                    except Exception as e:
                        logger.error('Failed to score transaction %s: %s', msg.get('transaction_id'), e)
            self.producer.send(results)
            n_fraud = sum(r['fraud_flag'] for r in results)
            logger.info('Scored %d transactions (%d flagged as fraud)', len(results), n_fraud)


if __name__ == '__main__':
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
    finally:
        service.consumer.close()
