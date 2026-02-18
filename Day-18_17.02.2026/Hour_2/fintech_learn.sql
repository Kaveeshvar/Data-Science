-- For the last 30 days in the dataset, show signup_date, signups.
WITH max_date AS(
    SELECT MAX(date(signup_ts)) as max_signup_date
    FROM users
)
SELECT date(u.signup_ts) AS signup_date,
    COUNT(*) AS signups
FROM users u
    CROSS JOIN max_date m
WHERE date(u.signup_ts) >= date(m.max_signup_date, '-30 days')
GROUP BY date(u.signup_ts)
ORDER BY signup_date;
-- What % of users are activated (have a non-null activation_ts)?
SELECT CASE
        WHEN COUNT(*) = 0 THEN 0
        ELSE 100.0 * SUM(activation_ts IS NOT NULL) / COUNT(*)
    END AS percent_activated
FROM users;
-- By signup week, compute:
-- signups
-- users who completed KYC within 7 days of signup (kyc_ts <= signup_ts + 7 days)
-- conversion rate
WITH cohorts AS (
    SELECT user_id,
        date(
            signup_ts,
            '-' || (
                (CAST(STRFTIME('%w', signup_ts) as integer) + 6) %7
            ) || ' days'
        ) AS signup_week,
        signup_ts,
        kyc_ts
    FROM users
    WHERE signup_ts IS NOT NULL
)
SELECT signup_week,
    COUNT(*) AS signups,
    SUM(
        CASE
            WHEN kyc_ts IS NOT NULL
            AND kyc_ts <= datetime(signup_ts, '+7 days') THEN 1
            ELSE 0
        END
    ) AS kyc_within_7d,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN kyc_ts IS NOT NULL
                AND kyc_ts <= datetime(signup_ts, '+7 days') THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS Conversion_percentage
FROM cohorts
GROUP BY signup_week
ORDER BY signup_week;
-- Payment success rate by acquisition channel
-- Question: For each acq_channel, compute:
-- total transactions
-- successful transactions
-- success rate
-- Tables: users, transactions
WITH total_txns_per_channel AS(
    SELECT u.acq_channel,
        txn_id,
        status
    FROM transactions t
        LEFT JOIN users u ON t.user_id = u.user_id
)
SELECT acq_channel,
    count(txn_id) as total_txns,
    SUM(
        CASE
            WHEN status = 'success' THEN 1
            ELSE 0
        END
    ) AS successful_txns,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN status = 'success' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS Success_precentage
FROM total_txns_per_channel
GROUP BY acq_channel;
-- Top merchants by GMV + success rate
-- Question: Show top 10 merchants by successful GMV (sum of amount where success), along with their success rate.
-- Tables: transactions, merchants
SELECT t.merchant_id,
    SUM(
        CASE
            WHEN UPPER(t.status) = 'SUCCESS' THEN t.amount
            ELSE 0
        END
    ) AS successful_gmv,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN UPPER(t.status) = 'SUCCESS' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS success_rate_pct
FROM transactions t
GROUP BY t.merchant_id
ORDER BY successful_gmv DESC
LIMIT 10;
-- Refund rate by MCC and tier
-- Question: For each mcc and merchant_tier, compute:
-- number of successful txns
-- number of refunded txns
-- refund rate
-- Tables: transactions, refunds, merchants
WITH refunded_txns AS(
    SELECT m.mcc,
        m.merchant_tier,
        COUNT(*) AS Refund_txns
    from refunds r
        LEFT JOIN transactions t ON r.txn_id = t.txn_id
        LEFT JOIN merchants m ON m.merchant_id = t.merchant_id
    GROUP BY m.mcc,
        m.merchant_tier
),
total_n_succ_txns AS(
    SELECT m.mcc,
        m.merchant_tier,
        COUNT(*) as total_txns,
        SUM(
            CASE
                WHEN UPPER(t.status) = 'SUCCESS' THEN 1
                ELSE 0
            END
        ) AS successful_txns
    FROM transactions t
        LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
    GROUP BY m.mcc,
        m.merchant_tier
)
SELECT *,
    ROUND(
        100.0 * Refund_txns / successful_txns,
        2
    ) AS refund_rate
FROM total_n_succ_txns t
    LEFT JOIN refunded_txns r ON t.mcc = r.mcc
    and t.merchant_tier = r.merchant_tier
ORDER BY refund_rate;
-- Contribution margin by channel
-- Question: For each acq_channel, compute:
-- net revenue = rev_interchange + rev_mdr_share
-- total costs = processing_fee + cashback_cost + refunded_amount (use refund_amount)
-- contribution margin = revenue - costs
-- Also output margin %.
-- Tables: users, transactions, refunds

WITH base AS(
    SELECT u.acq_channel,
        SUM(t.rev_interchange) + SUM(t.rev_mdr_share) AS net_revenue,
        (
            SUM(t.processing_fee) + SUM(t.cashback_cost) + SUM(r.refund_amount)
        ) as total_costs
    FROM users u
        LEFT JOIN transactions t ON u.user_id = t.user_id
        LEFT JOIN refunds r ON t.txn_id = r.txn_id
    WHERE t.status = 'success'
    GROUP BY u.acq_channel
)
SELECT *,
    net_revenue - total_costs as CM,
    ROUND(
        CASE
            WHEN net_revenue = 0 THEN 0
            ELSE 100.0 * (net_revenue - total_costs) / net_revenue
        END,
        2
    ) as margin_pct
FROM base;