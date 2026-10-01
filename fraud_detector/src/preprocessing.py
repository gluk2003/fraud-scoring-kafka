"""Препроцессинг транзакций для inference.

Все статистики, нужные для построения признаков (агрегаты по клиентам,
по категориям и т.д.), рассчитываются заранее скриптом training/train.py
и сохраняются в лёгкий JSON-файл артефактов. Сервису не нужен train.csv.
"""
import hashlib
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CAT_FEATURES = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']

NUM_FEATURES = [
    'amount', 'amount_log',
    'hour', 'day_of_week', 'day_of_month', 'month', 'is_night',
    'distance_km', 'population_log',
    'client_txn_count', 'client_amount_mean', 'client_amount_std',
    'amount_to_client_mean', 'amount_client_zscore',
    'cat_amount_mean', 'amount_to_cat_mean',
]

FEATURES = CAT_FEATURES + NUM_FEATURES

REQUIRED_COLUMNS = [
    'transaction_time', 'merch', 'cat_id', 'amount', 'name_1', 'name_2',
    'gender', 'street', 'one_city', 'us_state', 'lat', 'lon',
    'population_city', 'jobs', 'merchant_lat', 'merchant_lon',
]


def client_key(df: pd.DataFrame) -> pd.Series:
    """Идентификатор клиента: хэш от имени, фамилии и адреса."""
    raw = df['name_1'].astype(str) + '|' + df['name_2'].astype(str) + '|' + df['street'].astype(str)
    return raw.map(lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()[:16])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_artifacts(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        artifacts = json.load(f)
    logger.info('Preprocessing artifacts loaded: %d clients, %d categories',
                len(artifacts['client_stats']), len(artifacts['cat_amount_mean']))
    return artifacts


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def run_preproc(input_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Сырые транзакции формата test.csv -> матрица признаков модели."""
    validate_input(input_df)
    df = input_df.copy()
    out = pd.DataFrame(index=df.index)

    # Категориальные признаки CatBoost принимает строками
    for col in CAT_FEATURES:
        out[col] = df[col].fillna('NA').astype(str)

    # Сумма транзакции
    amount = pd.to_numeric(df['amount'], errors='coerce').fillna(artifacts['global_amount_mean'])
    out['amount'] = amount
    out['amount_log'] = np.log1p(amount.clip(lower=0))

    # Временные признаки
    ts = pd.to_datetime(df['transaction_time'], errors='coerce')
    out['hour'] = ts.dt.hour.fillna(12).astype(int)
    out['day_of_week'] = ts.dt.dayofweek.fillna(0).astype(int)
    out['day_of_month'] = ts.dt.day.fillna(15).astype(int)
    out['month'] = ts.dt.month.fillna(6).astype(int)
    out['is_night'] = ((out['hour'] >= 22) | (out['hour'] < 6)).astype(int)

    # Гео: расстояние клиент <-> мерчант
    coords = df[['lat', 'lon', 'merchant_lat', 'merchant_lon']].apply(pd.to_numeric, errors='coerce')
    out['distance_km'] = haversine_km(coords['lat'], coords['lon'],
                                      coords['merchant_lat'], coords['merchant_lon'])
    out['distance_km'] = out['distance_km'].fillna(artifacts['global_distance_mean'])

    population = pd.to_numeric(df['population_city'], errors='coerce').fillna(artifacts['global_population_median'])
    out['population_log'] = np.log1p(population.clip(lower=0))

    # Профиль клиента по истории транзакций (рассчитан на train)
    stats = pd.DataFrame.from_dict(artifacts['client_stats'], orient='index',
                                   columns=['client_txn_count', 'client_amount_mean', 'client_amount_std'])
    keys = client_key(df)
    client = stats.reindex(keys.values).set_index(df.index)
    client['client_txn_count'] = client['client_txn_count'].fillna(0)
    client['client_amount_mean'] = client['client_amount_mean'].fillna(artifacts['global_amount_mean'])
    client['client_amount_std'] = client['client_amount_std'].fillna(artifacts['global_amount_std'])
    out = out.join(client)

    out['amount_to_client_mean'] = amount / (out['client_amount_mean'] + 1)
    out['amount_client_zscore'] = (amount - out['client_amount_mean']) / (out['client_amount_std'] + 1)

    # Средний чек по категории покупки
    cat_mean = df['cat_id'].astype(str).map(artifacts['cat_amount_mean']).astype(float)
    out['cat_amount_mean'] = cat_mean.fillna(artifacts['global_amount_mean'])
    out['amount_to_cat_mean'] = amount / (out['cat_amount_mean'] + 1)

    return out[FEATURES]
