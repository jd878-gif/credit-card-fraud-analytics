-- =====================================================================
-- 03_analysis.sql — business questions
-- =====================================================================

-- Q1. Which categories have the highest fraud rate?
SELECT category,
       COUNT(*)                                   AS transactions,
       SUM(is_fraud)                              AS fraud_count,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY category
ORDER BY fraud_rate_pct DESC;

-- Q2. Which hours of the day are riskiest?
SELECT EXTRACT(HOUR FROM trans_date_trans_time)::int AS hour,
       COUNT(*)                                      AS transactions,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2)    AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY fraud_rate_pct DESC;

-- Q3. Which states lose the most money to fraud?
SELECT state,
       SUM(is_fraud)                            AS fraud_count,
       ROUND(SUM(amt) FILTER (WHERE is_fraud = 1), 0) AS fraud_amount
FROM transactions
GROUP BY state
ORDER BY fraud_amount DESC NULLS LAST
LIMIT 10;

-- Q4. Monthly spend and fraud, with month-over-month change (window function)
WITH monthly AS (
    SELECT DATE_TRUNC('month', trans_date_trans_time)::date AS month,
           SUM(amt)       AS total_spend,
           SUM(is_fraud)  AS fraud_count
    FROM transactions
    GROUP BY 1
)
SELECT month,
       ROUND(total_spend, 0) AS total_spend,
       fraud_count,
       ROUND((total_spend - LAG(total_spend) OVER (ORDER BY month)) * 100.0
             / NULLIF(LAG(total_spend) OVER (ORDER BY month), 0), 1) AS spend_mom_pct
FROM monthly
ORDER BY month;

-- Q5. Fraud rate by gender
SELECT gender,
       COUNT(*)                                   AS transactions,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY gender;

-- Q6. Top 10 merchants by fraud count (minimum volume to avoid noise)
SELECT merchant,
       COUNT(*)                                   AS transactions,
       SUM(is_fraud)                              AS fraud_count,
       ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY merchant
HAVING COUNT(*) >= 500
ORDER BY fraud_count DESC
LIMIT 10;
