-- Витрина результатов скоринга (выполняется при первом старте контейнера postgres)
CREATE TABLE IF NOT EXISTS scores (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64)      NOT NULL UNIQUE,
    score          DOUBLE PRECISION NOT NULL,
    fraud_flag     SMALLINT         NOT NULL CHECK (fraud_flag IN (0, 1)),
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scores_created_at ON scores (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scores_fraud_created_at ON scores (created_at DESC) WHERE fraud_flag = 1;
