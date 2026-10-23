-- =====================================================================
-- 01_schema.sql — table definition and data load
-- Dataset: Kaggle "Credit Card Transactions Dataset" (priyamchoksi),
--          a simulated dataset generated with the Sparkov tool.
-- Run:     psql -d credit_card_db -f sql/01_schema.sql
-- =====================================================================

DROP TABLE IF EXISTS transactions CASCADE;

CREATE TABLE transactions (
    row_id                 INTEGER,          -- unnamed index column in the CSV
    trans_date_trans_time  TIMESTAMP     NOT NULL,
    cc_num                 BIGINT        NOT NULL,
    merchant               TEXT          NOT NULL,
    category               TEXT          NOT NULL,
    amt                    NUMERIC(10,2) NOT NULL CHECK (amt >= 0),
    first                  TEXT,
    last                   TEXT,
    gender                 CHAR(1),
    street                 TEXT,
    city                   TEXT,
    state                  CHAR(2),
    zip                    INTEGER,
    lat                    DOUBLE PRECISION,
    long                   DOUBLE PRECISION,
    city_pop               INTEGER,
    job                    TEXT,
    dob                    DATE,
    trans_num              TEXT          PRIMARY KEY,
    unix_time              BIGINT,
    merch_lat              DOUBLE PRECISION,
    merch_long             DOUBLE PRECISION,
    is_fraud               SMALLINT      NOT NULL CHECK (is_fraud IN (0, 1)),
    merch_zipcode          TEXT
);

-- Load the CSV (psql client-side copy; adjust the path to your download):
-- \copy transactions FROM 'data/credit_card_transactions.csv' WITH (FORMAT csv, HEADER true)
