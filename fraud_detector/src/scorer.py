"""Скоринг подготовленных признаков предобученной моделью CatBoost (CPU)."""
import json
import logging

import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, model_path: str, meta_path: str):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        self.threshold = float(meta['threshold'])
        self.features = meta['features']
        logger.info('Model loaded from %s, threshold=%.4f', model_path, self.threshold)

    def make_pred(self, features: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(features[self.features])[:, 1]
        return pd.DataFrame({
            'score': proba.round(6),
            'fraud_flag': (proba >= self.threshold).astype(int),
        }, index=features.index)
