SELECT SUM(amount) FROM fact_transactions;


-- Describe Table
PRAGMA table_info(fact_transactions);


-- Assign each user to a cohort month based on first successful transaction.
WITH first_success AS (
  SELECT
    user_id,
    strftime('%Y-%m-01', MIN(created_ts)) AS cohort_month
  FROM fact_transactions
  WHERE status = 'success'
  GROUP BY user_id
)
SELECT 
  cohort_month,
  COUNT(*) AS users_in_cohort
FROM first_success
GROUP BY cohort_month
ORDER BY cohort_month;


-- 30-day retention calculation with window functions
-- User is retained if they have ≥1 successful txn in days 1–30 after first success.
WITH tx AS (
  SELECT
    user_id,
    created_ts,
    MIN(CASE WHEN status = 'success' THEN created_ts END)
      OVER (PARTITION BY user_id) AS first_success_ts,
    status
  FROM fact_transactions
),
retention AS (
  SELECT
    user_id,
    first_success_ts,
    MAX(
      CASE
        WHEN status = 'success'
         AND created_ts > first_success_ts
         AND created_ts <= datetime(first_success_ts, '+30 days')
        THEN 1 ELSE 0
      END
    ) AS retained_30d
  FROM tx
  WHERE first_success_ts IS NOT NULL
  GROUP BY user_id, first_success_ts
)
SELECT
  strftime('%Y-%m-01', first_success_ts) AS cohort_month,
  COUNT(*) AS activated_users,
  SUM(retained_30d) AS retained_users_30d,
  1.0 * SUM(retained_30d) / COUNT(*) AS retention_30d
FROM retention
GROUP BY cohort_month
ORDER BY cohort_month;



-- Funnel conversion query (signup → KYC → first txn)
-- One row per user with milestone timestamps, then compute conversion.
WITH kyc AS (
  SELECT
    user_id,
    MIN(kyc_ts) AS kyc_approved_ts
  FROM fact_kyc_events
  WHERE status = 'completed'
  GROUP BY user_id
),
first_txn AS (
  SELECT
    user_id,
    MIN(created_ts) AS first_success_ts
  FROM fact_transactions
  WHERE status = 'success'
  GROUP BY user_id
),
funnel AS (
  SELECT
    u.user_id,
    u.created_ts AS signup_ts,
    k.kyc_approved_ts,
    t.first_success_ts
  FROM dim_user u
  LEFT JOIN kyc k ON u.user_id = k.user_id
  LEFT JOIN first_txn t ON u.user_id = t.user_id
)
SELECT
  date(signup_ts, 'weekday 1', '-7 days') AS signup_week,
  COUNT(*) AS signups,
  SUM(
    CASE
      WHEN kyc_approved_ts IS NOT NULL
       AND kyc_approved_ts <= datetime(signup_ts, '+7 days')
      THEN 1 ELSE 0
    END
  ) AS kyc_7d,
  SUM(
    CASE
      WHEN first_success_ts IS NOT NULL
       AND first_success_ts <= datetime(signup_ts, '+7 days')
      THEN 1 ELSE 0
    END
  ) AS first_txn_7d,
  1.0 * SUM(
    CASE
      WHEN kyc_approved_ts IS NOT NULL
       AND kyc_approved_ts <= datetime(signup_ts, '+7 days')
      THEN 1 ELSE 0
    END
  ) / COUNT(*) AS signup_to_kyc_7d,
  1.0 * SUM(
    CASE
      WHEN first_success_ts IS NOT NULL
       AND first_success_ts <= datetime(signup_ts, '+7 days')
      THEN 1 ELSE 0
    END
  ) / COUNT(*) AS signup_to_txn_7d
FROM funnel
GROUP BY signup_week
ORDER BY signup_week;


