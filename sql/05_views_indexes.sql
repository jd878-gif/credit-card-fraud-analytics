-- =====================================================================
-- 05_views_indexes.sql — indexes and reporting views
-- =====================================================================

-- Indexes for the access patterns used above
CREATE INDEX IF NOT EXISTS idx_tx_card_time ON transactions (cc_num, trans_date_trans_time);
CREATE INDEX IF NOT EXISTS idx_tx_category  ON transactions (category);
CREATE INDEX IF NOT EXISTS idx_tx_time      ON transactions (trans_date_trans_time);
-- Partial index: fraud investigations only scan the ~0.6% fraud rows
CREATE INDEX IF NOT EXISTS idx_tx_fraud     ON transactions (trans_date_trans_time) WHERE is_fraud = 1;

ANALYZE transactions;

-- Daily KPI view for dashboards
CREATE OR REPLACE VIEW v_daily_fraud_summary AS
SELECT trans_date_trans_time::date                  AS day,
       COUNT(*)                                     AS transactions,
       SUM(is_fraud)                                AS fraud_count,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 3)   AS fraud_rate_pct,
       ROUND(SUM(amt) FILTER (WHERE is_fraud = 1), 2) AS fraud_amount
FROM transactions
GROUP BY 1;

-- Category risk view
CREATE OR REPLACE VIEW v_category_risk AS
SELECT category,
       COUNT(*)                                     AS transactions,
       SUM(is_fraud)                                AS fraud_count,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)   AS fraud_rate_pct,
       ROUND(AVG(amt) FILTER (WHERE is_fraud = 1), 2) AS avg_fraud_amt
FROM transactions
GROUP BY category;

-- Materialized view: per-card profile (refresh after each load)
DROP MATERIALIZED VIEW IF EXISTS mv_card_profile;
CREATE MATERIALIZED VIEW mv_card_profile AS
SELECT cc_num,
       COUNT(*)                     AS transactions,
       ROUND(AVG(amt), 2)           AS avg_amt,
       MAX(trans_date_trans_time)   AS last_seen,
       SUM(is_fraud)                AS fraud_count
FROM transactions
GROUP BY cc_num;
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_card_profile ON mv_card_profile (cc_num);

-- Check an index is used (example)
EXPLAIN SELECT * FROM transactions
WHERE cc_num = 2703186189652095 ORDER BY trans_date_trans_time;

-- REFRESH MATERIALIZED VIEW CONCURRENTLY mv_card_profile;
