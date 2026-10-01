# Real-Time Fraud Scoring (Kafka + CatBoost + PostgreSQL)

Сервис потокового скоринга фродовых транзакций. Транзакции формата `test.csv`
из соревнования [teta-ml-1-2025](https://www.kaggle.com/competitions/teta-ml-1-2025)
читаются из топика Kafka `transactions`, проходят препроцессинг и скоринг моделью
CatBoost (только inference, только CPU), а результат (`transaction_id`, `score`,
`fraud_flag`) пишется в топик `scores`. Отдельный сервис складывает результаты
в PostgreSQL, а в UI есть раздел с просмотром результатов.

Проект основан на [коде с семинара](https://github.com/NikitaMalykhin/mts25_mlops_hw2_real_time_fraud_detection).

## Архитектура

```
 ┌───────────┐  transactions  ┌────────────────┐    scores    ┌───────────┐       ┌────────────┐
 │ interface │ ─────────────► │ fraud_detector │ ───────────► │ db_writer │ ────► │ PostgreSQL │
 │(Streamlit)│     Kafka      │ preproc+model  │    Kafka     │           │       │   scores   │
 └─────┬─────┘                └────────────────┘              └───────────┘       └─────┬──────┘
       └──────────────────────── «Посмотреть результаты» (SELECT) ◄─────────────────────┘
```

| Сервис | Что делает | Порт на хосте |
|---|---|---|
| `zookeeper`, `kafka` | брокер сообщений | 9095 (внешний listener) |
| `kafka-setup` | создаёт топики `transactions` и `scores` (3 партиции) и завершается | — |
| `kafka-ui` | просмотр топиков и сообщений | [8080](http://localhost:8080) |
| `postgres` | БД `fraud`, витрина `scores` создаётся [postgres/init.sql](postgres/init.sql) | 5433 |
| `fraud_detector` | ML-сервис: Kafka → препроцессинг → модель → Kafka | — |
| `db_writer` | читает топик `scores` и пишет в таблицу `scores` | — |
| `interface` | Streamlit UI: отправка CSV в Kafka и раздел «Результаты» | [8501](http://localhost:8501) |

### Сервис скоринга `fraud_detector`

Этапы ML-пайплайна вынесены в отдельные скрипты:

| Скрипт | Этап |
|---|---|
| [src/kafka_io.py](fraud_detector/src/kafka_io.py) | чтение сообщений из топика `transactions` и запись скора и флага фрода в топик `scores` |
| [src/preprocessing.py](fraud_detector/src/preprocessing.py) | препроцессинг сырой транзакции в признаки модели |
| [src/scorer.py](fraud_detector/src/scorer.py) | скоринг признаков моделью CatBoost, применение порога |
| [app/app.py](fraud_detector/app/app.py) | точка входа: связывает этапы в цикл обработки |

Сообщения читаются пачками (до 500 штук), скорятся векторно; если пачка падает
из-за битой записи, сервис скорит сообщения по одному, чтобы ошибка одной
транзакции не блокировала остальные. Логи: `docker compose logs fraud_detector`
или файл `/app/logs/service.log` внутри контейнера.

**Формат входного сообщения** (`transactions`) — как в семинаре:
```json
{"transaction_id": "8f0c…", "data": {"transaction_time": "2019-09-14 02:46", "merch": "…", "amount": 25.79, "...": "все колонки test.csv"}}
```

**Формат выходного сообщения** (`scores`):
```json
{"transaction_id": "8f0c…", "score": 0.000071, "fraud_flag": 0}
```

### Модель и признаки

Модель обучается офлайн скриптом [training/train.py](training/train.py), контейнер
только загружает готовые артефакты из [fraud_detector/models/](fraud_detector/models):

- `model.cbm` — CatBoost (≈800 КБ, `max_ctr_complexity=1` для облегчения модели);
- `preprocessing_artifacts.json` — статистики, посчитанные на train (≈45 КБ).
  Благодаря им в контейнер не нужно тащить `train.csv` (145 МБ), как в семинаре;
- `model_meta.json` — список признаков, порог и метрики на валидации.

Признаки:
- категориальные (нативно в CatBoost): `gender`, `merch`, `cat_id`, `one_city`, `us_state`, `jobs`;
- сумма: `amount`, `log(1+amount)`;
- время: час, день недели, день месяца, месяц, флаг ночной транзакции;
- гео: расстояние клиент ↔ мерчант (haversine), `log(1+population_city)`;
- профиль клиента (клиент = хэш имя+фамилия+адрес): число транзакций, средний чек
  и std в train, отношение суммы к среднему чеку клиента, z-score суммы;
- средний чек по категории `cat_id` и отношение суммы к нему.

Порог `fraud_flag` подобран по максимуму F1 на валидации (20% train, стратифицированно).

| Метрика (валидация) | Значение |
|---|---|
| ROC-AUC | 0.9984 |
| F1 | 0.856 |
| Precision / Recall | 0.923 / 0.798 |
| Порог | 0.818 |

Переобучить модель (необязательно, готовые артефакты уже лежат в репозитории):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
python training/train.py --train /path/to/train.csv   # ~3 минуты на CPU
```

## Быстрый старт

### Требования
- Docker 20.10+ и Docker Compose v2 (`docker compose`)
- Свободные порты 8080, 8501, 9095, 5433

### Запуск
```bash
git clone https://github.com/gluk2003/fraud-scoring-kafka.git
cd fraud-scoring-kafka
docker compose up --build -d
```

Первый запуск занимает несколько минут (скачивание образов и сборка). Проверить, что всё поднялось:
```bash
docker compose ps
```
Сервисы `fraud_detector`, `db_writer`, `interface`, `kafka`, `postgres`, `kafka-ui`
должны быть в статусе `running`, `kafka-setup` — `exited (0)` (он только создаёт топики).

## Проверка работоспособности

1. Откройте UI: http://localhost:8501, вкладка **«📤 Отправка транзакций»**.
2. Загрузите CSV формата `test.csv`. Для быстрого теста в репозитории есть:
   - [sample_data/test_sample_100.csv](sample_data/test_sample_100.csv) — 100 случайных транзакций из `test.csv`;
   - [sample_data/test_sample_1000.csv](sample_data/test_sample_1000.csv) — 1000 транзакций,
     среди которых гарантированно есть транзакции, которые модель помечает как фрод (~40 шт.).

   Можно загрузить и полный `test.csv` из соревнования (262 144 транзакции) — сервис скорит его пачками.
3. Нажмите **«Отправить …»**.
4. Перейдите на вкладку **«📈 Результаты»** и нажмите **«Посмотреть результаты»**. Вы увидите:
   - число транзакций и фродов в базе;
   - 10 последних транзакций с `fraud_flag = 1`;
   - гистограмму распределения скоров последних 100 транзакций.

Дополнительно:
- **Kafka UI** — http://localhost:8080 → Topics → `transactions` / `scores` → Messages.
- **Логи сервисов**:
  ```bash
  docker compose logs -f fraud_detector   # Scored 500 transactions (21 flagged as fraud)
  docker compose logs -f db_writer        # Saved 500 scores to PostgreSQL
  ```
- **Витрина в PostgreSQL**:
  ```bash
  docker compose exec postgres psql -U fraud -d fraud \
    -c "SELECT count(*), sum(fraud_flag) FROM scores;" \
    -c "SELECT * FROM scores ORDER BY created_at DESC LIMIT 5;"
  ```
  С хоста можно подключиться к `localhost:5433`, БД/пользователь/пароль — `fraud`.

### Остановка
```bash
docker compose down        # остановить контейнеры
docker compose down -v     # + удалить volume с данными PostgreSQL
```

## Витрина `scores` в PostgreSQL

```sql
CREATE TABLE scores (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64)      NOT NULL UNIQUE,
    score          DOUBLE PRECISION NOT NULL,
    fraud_flag     SMALLINT         NOT NULL,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);
```

`db_writer` вставляет записи пачками через `INSERT … ON CONFLICT (transaction_id) DO UPDATE`
и коммитит offset Kafka только после успешной записи в БД, поэтому при рестарте
сообщения не теряются и не дублируются.

## Структура проекта

```
.
├── docker-compose.yml
├── fraud_detector/            # ML-сервис скоринга
│   ├── app/app.py             # точка входа, цикл обработки
│   ├── src/kafka_io.py        # Kafka consumer/producer
│   ├── src/preprocessing.py   # препроцессинг
│   ├── src/scorer.py          # inference CatBoost
│   ├── models/                # model.cbm, model_meta.json, preprocessing_artifacts.json
│   ├── Dockerfile
│   └── requirements.txt
├── db_writer/                 # Kafka (scores) -> PostgreSQL
│   ├── app/app.py
│   ├── Dockerfile
│   └── requirements.txt
├── interface/                 # Streamlit UI
│   ├── app.py
│   ├── .streamlit/config.toml
│   ├── Dockerfile
│   └── requirements.txt
├── postgres/init.sql          # создание витрины scores
├── training/                  # офлайн-обучение модели (вне контейнеров)
│   ├── train.py
│   └── requirements.txt
└── sample_data/               # сэмплы test.csv для проверки
```
