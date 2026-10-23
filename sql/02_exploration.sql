-- =====================================================================
-- 02_exploration.sql — data profiling and quality checks
-- =====================================================================

-- 1. Size, date range, distinct entities
SELECT COUNT(*)                         AS total_transactions,
       MIN(trans_date_trans_time)::date AS first_day,
       MAX(trans_date_trans_time)::date AS last_day,
       COUNT(DISTINCT cc_num)           AS cards,
       COUNT(DISTINCT merchant)         AS merchants
FROM transactions;

-- 2. Class balance: share of fraud vs legitimate
SELECT is_fraud,
       COUNT(*)                                            AS total,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)  AS percentage
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;

-- 3. Null checks on columns used in the analysis
SELECT COUNT(*) FILTER (WHERE amt IS NULL)                   AS null_amt,
       COUNT(*) FILTER (WHERE category IS NULL)              AS null_category,
       COUNT(*) FILTER (WHERE trans_date_trans_time IS NULL) AS null_timestamp,
       COUNT(*) FILTER (WHERE lat IS NULL OR merch_lat IS NULL) AS null_coordinates
FROM transactions;

-- 4. Duplicate transaction IDs (should be 0; trans_num is the primary key)
SELECT COUNT(*) - COUNT(DISTINCT trans_num) AS duplicate_trans_num
FROM transactions;

-- 5. Amount distribution, fraud vs legitimate
SELECT is_fraud,
       ROUND(AVG(amt), 2)                                              AS mean_amt,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amt)::numeric, 2) AS median_amt,
       ROUND(STDDEV(amt), 2)                                           AS std_amt,
       MIN(amt)                                                        AS min_amt,
       MAX(amt)                                                        AS max_amt
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;

-- 6. Transactions per category
SELECT category, COUNT(*) AS transactions
FROM transactions
GROUP BY category
ORDER BY transactions DESC;
