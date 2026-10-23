-- =====================================================================
-- 04_fraud_detection.sql — rule-based detection in SQL
-- =====================================================================

-- 1. Velocity check: card used again within 10 minutes (LAG over each card)
WITH ordered AS (
    SELECT trans_num, cc_num, trans_date_trans_time, amt, is_fraud,
           trans_date_trans_time
             - LAG(trans_date_trans_time) OVER (PARTITION BY cc_num
                                                ORDER BY trans_date_trans_time) AS gap
    FROM transactions
)
SELECT (gap <= INTERVAL '10 minutes')            AS within_10_min,
       COUNT(*)                                  AS transactions,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM ordered
WHERE gap IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- 2. Spend vs the card's OWN HISTORY (prior transactions only, no look-ahead)
WITH history AS (
    SELECT trans_num, amt, is_fraud,
           AVG(amt) OVER (PARTITION BY cc_num
                          ORDER BY trans_date_trans_time
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_avg
    FROM transactions
)
SELECT CASE WHEN prior_avg IS NULL      THEN 'first transaction'
            WHEN amt > 5 * prior_avg    THEN '> 5x usual spend'
            WHEN amt > 2 * prior_avg    THEN '2-5x usual spend'
            ELSE 'normal' END                   AS spend_vs_history,
       COUNT(*)                                 AS transactions,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM history
GROUP BY 1
ORDER BY fraud_rate_pct DESC;

-- 3. Multi-factor risk score (amount + category + time of day), max 7
--    Uses only information available when the transaction happens.
CREATE OR REPLACE VIEW v_risk_scores AS
SELECT trans_num, cc_num, trans_date_trans_time, amt, category, merchant, is_fraud,
       (CASE WHEN amt > 1000 THEN 3 WHEN amt > 500 THEN 2 WHEN amt > 200 THEN 1 ELSE 0 END)
     + (CASE WHEN category IN ('shopping_net', 'misc_net', 'grocery_pos') THEN 2 ELSE 0 END)
     + (CASE WHEN EXTRACT(HOUR FROM trans_date_trans_time) >= 22
               OR EXTRACT(HOUR FROM trans_date_trans_time) <= 3 THEN 2 ELSE 0 END) AS fraud_score
FROM transactions;

-- 4. How well does the score work? Confusion counts per risk level
SELECT CASE WHEN fraud_score >= 5 THEN 'High Risk'
            WHEN fraud_score >= 3 THEN 'Medium Risk'
            ELSE 'Low Risk' END                  AS risk_level,
       COUNT(*)                                  AS flagged,
       SUM(is_fraud)                             AS actual_fraud,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS precision_pct,
       ROUND(SUM(is_fraud) * 100.0
             / (SELECT SUM(is_fraud) FROM transactions), 1) AS share_of_all_fraud_pct
FROM v_risk_scores
GROUP BY 1
ORDER BY MIN(fraud_score) DESC;

-- 5. Top 20 highest-risk transactions for analyst review
SELECT trans_num, trans_date_trans_time, amt, category, merchant, fraud_score, is_fraud
FROM v_risk_scores
ORDER BY fraud_score DESC, amt DESC
LIMIT 20;
