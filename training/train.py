"""Офлайн-обучение модели и расчёт артефактов препроцессинга.

Запускается вне контейнеров (контейнер делает только inference):
    python training/train.py --train path/to/train.csv

Результат кладётся в fraud_detector/models/:
    model.cbm                     - модель CatBoost
    model_meta.json               - порог, список признаков, метрики
    preprocessing_artifacts.json  - статистики для построения признаков
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, 'fraud_detector', 'src'))
from preprocessing import CAT_FEATURES, FEATURES, client_key, haversine_km, run_preproc  # noqa: E402

RANDOM_STATE = 42


def fit_artifacts(train: pd.DataFrame) -> dict:
    keys = client_key(train)
    client_stats = train.groupby(keys)['amount'].agg(['count', 'mean', 'std']).fillna(0)
    distance = haversine_km(train['lat'], train['lon'], train['merchant_lat'], train['merchant_lon'])
    return {
        'client_stats': {k: [int(r['count']), round(float(r['mean']), 4), round(float(r['std']), 4)]
                         for k, r in client_stats.iterrows()},
        'cat_amount_mean': {k: round(float(v), 4) for k, v in train.groupby('cat_id')['amount'].mean().items()},
        'global_amount_mean': float(train['amount'].mean()),
        'global_amount_std': float(train['amount'].std()),
        'global_distance_mean': float(distance.mean()),
        'global_population_median': float(train['population_city'].median()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=os.path.join(ROOT, '..', 'data', 'train.csv'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'fraud_detector', 'models'))
    parser.add_argument('--iterations', type=int, default=600)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    print('Train shape:', train.shape, 'fraud rate:', round(train['target'].mean(), 5))

    tr, val = train_test_split(train, test_size=0.2, stratify=train['target'], random_state=RANDOM_STATE)

    # Валидация: артефакты считаем только на трейн-части, чтобы не было утечки
    art_tr = fit_artifacts(tr)
    X_tr, X_val = run_preproc(tr, art_tr), run_preproc(val, art_tr)
    params = dict(iterations=args.iterations, depth=6, learning_rate=0.08, eval_metric='AUC',
                  auto_class_weights='SqrtBalanced', max_ctr_complexity=1, random_seed=RANDOM_STATE, task_type='CPU', verbose=100)
    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, tr['target'], cat_features=CAT_FEATURES),
              eval_set=Pool(X_val, val['target'], cat_features=CAT_FEATURES))

    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(val['target'], proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best])
    metrics = {
        'val_roc_auc': round(float(roc_auc_score(val['target'], proba)), 5),
        'val_f1': round(float(f1_score(val['target'], proba >= threshold)), 5),
        'val_precision': round(float(precision[best]), 5),
        'val_recall': round(float(recall[best]), 5),
    }
    print('Validation metrics:', metrics, 'threshold:', round(threshold, 4))

    # Финальная модель: обучаем на всём train с тем же числом итераций
    artifacts = fit_artifacts(train)
    final = CatBoostClassifier(**{**params, 'iterations': model.get_best_iteration() + 1, 'verbose': 200})
    final.fit(Pool(run_preproc(train, artifacts), train['target'], cat_features=CAT_FEATURES))

    os.makedirs(args.out, exist_ok=True)
    final.save_model(os.path.join(args.out, 'model.cbm'))
    with open(os.path.join(args.out, 'preprocessing_artifacts.json'), 'w', encoding='utf-8') as f:
        json.dump(artifacts, f)
    with open(os.path.join(args.out, 'model_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'threshold': threshold, 'features': FEATURES, 'cat_features': CAT_FEATURES,
                   'metrics': metrics}, f, indent=2, ensure_ascii=False)
    print('Saved model and artifacts to', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
