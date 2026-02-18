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
-- CAC + Payback window (7d) by channel
-- Question: Compute CAC for each channel using:
-- Spend: marketing_spend.spend
-- Acquired users: users whose acq_channel = channel
-- Then compute 7-day revenue per acquired user and estimate payback:
-- cac = spend / signups
-- rev_7d_per_user = revenue_from_txns_in_7d_after_signup / signups
-- payback_ratio_7d = rev_7d_per_user / cac
-- Tables: marketing_spend, users, transactions
-- Total Spend per channel
-- ROUGH work
SELECT channel,
    SUM(spend)
FROM marketing_spend
GROUP BY channel;
-- Total activated signups per acq_channel
SELECT acq_channel,
    COUNT(user_id)
FROM users
WHERE activation_ts IS NOT NULL
    AND date(activation_ts) >= date(signup_ts, '+7 days')
GROUP BY acq_channel;
-- Total revenue from txns per user in signup+7days 
SELECT u.user_id,
    u.acq_channel,
    COALESCE(
        SUM(t.rev_interchange + t.rev_mdr_share),
        0
    ) AS revenue
FROM users u
    LEFT JOIN transactions t ON t.user_id = u.user_id
    AND t.txn_ts <= datetime(u.signup_ts, '+7 days')
GROUP BY u.user_id,
    u.acq_channel;
-- Channel-user_count-spend
WITH spend_per_ch AS (
    SELECT channel,
        SUM(spend) as channel_spend
    FROM marketing_spend
    GROUP BY channel
)
SELECT u.acq_channel,
    COUNT(u.user_id) AS signups,
    s.channel_spend AS spend_per_channel,
    s.channel_spend / COUNT(u.user_id) AS CAC
FROM users u
    RIGHT JOIN spend_per_ch s ON s.channel = u.acq_channel
WHERE u.activation_ts IS NOT NULL
    AND date(u.activation_ts) >= date(u.signup_ts, '+7 days')
GROUP BY u.acq_channel;
-- 7d revenue per channel
SELECT u.acq_channel,
    COALESCE(
        SUM(t.rev_interchange + t.rev_mdr_share),
        0
    ) AS revenue
FROM users u
    LEFT JOIN transactions t ON t.user_id = u.user_id
    AND t.txn_ts <= datetime(u.signup_ts, '+7 days')
GROUP BY u.acq_channel;
-- CAC + Payback window (7d) by channel
-- Question: Compute CAC for each channel using:
-- Spend: marketing_spend.spend
-- Acquired users: users whose acq_channel = channel
-- Then compute 7-day revenue per acquired user and estimate payback:
-- cac = spend / signups
-- rev_7d_per_user = revenue_from_txns_in_7d_after_signup / signups
-- payback_ratio_7d = rev_7d_per_user / cac
-- Tables: marketing_spend, users, transactions
-- Total Spend per channel
WITH spend_per_ch AS (
    SELECT channel,
        SUM(spend) as channel_spend
    FROM marketing_spend
    GROUP BY channel
),
rev_7d_per_channel AS(
    SELECT u.acq_channel AS channel,
        COALESCE(
            SUM(t.rev_interchange + t.rev_mdr_share),
            0
        ) AS revenue
    FROM users u
        LEFT JOIN transactions t ON t.user_id = u.user_id
        AND t.txn_ts <= datetime(u.signup_ts, '+7 days')
    GROUP BY u.acq_channel
)
SELECT u.acq_channel,
    COUNT(u.user_id) AS signups,
    s.channel_spend AS spend_per_channel,
    s.channel_spend / COUNT(u.user_id) AS CAC,
    r.revenue / COUNT(u.user_id) as rev_7d_per_user,
    100 *(
        (r.revenue / COUNT(u.user_id)) / (s.channel_spend / COUNT(u.user_id))
    ) AS payback_ratio_7d
FROM users u
    RIGHT JOIN spend_per_ch s ON s.channel = u.acq_channel
    RIGHT JOIN rev_7d_per_channel r ON r.channel = u.acq_channel
WHERE u.activation_ts IS NOT NULL
    AND date(u.activation_ts) >= date(u.signup_ts, '+7 days')
GROUP BY u.acq_channel;
-- Corrected query
-- CAC + Payback window (7d) by channel
WITH spend_per_ch AS (
    SELECT channel,
        SUM(spend) as channel_spend
    FROM marketing_spend
    GROUP BY channel
),
signups_per_channel AS (
    SELECT acq_channel AS channel,
        COUNT(user_id) AS signups
    FROM users
    GROUP BY acq_channel
),
rev_7d_per_channel AS (
    SELECT u.acq_channel AS channel,
        COALESCE(SUM(t.rev_interchange + t.rev_mdr_share), 0) AS revenue_7d
    FROM users u
        LEFT JOIN transactions t ON t.user_id = u.user_id
        AND t.txn_ts >= u.signup_ts
        AND t.txn_ts < datetime(u.signup_ts, '+7 days')
        AND UPPER(t.status) = 'SUCCESS'
    GROUP BY u.acq_channel
)
SELECT s.channel,
    COALESCE(u.signups, 0) AS signups,
    s.channel_spend AS spend_per_channel,
    CASE
        WHEN COALESCE(u.signups, 0) = 0 THEN NULL
        ELSE 1.0 * s.channel_spend / u.signups
    END AS CAC,
    CASE
        WHEN COALESCE(u.signups, 0) = 0 THEN NULL
        ELSE 1.0 * COALESCE(r.revenue_7d, 0) / u.signups
    END AS rev_7d_per_user,
    CASE
        WHEN s.channel_spend = 0 THEN NULL
        ELSE ROUND(
            100.0 * COALESCE(r.revenue_7d, 0) / s.channel_spend,
            2
        )
    END AS payback_ratio_7d
FROM spend_per_ch s
    LEFT JOIN signups_per_channel u ON u.channel = s.channel
    LEFT JOIN rev_7d_per_channel r ON r.channel = s.channel
ORDER BY payback_ratio_7d DESC;
-- D7 retention (txn-based)
-- Question: Define retention as: a user is “retained D7” if they have any successful transaction between day 7 and day 14 after their first successful transaction.
-- Compute by activation week:
-- activated users
-- retained users
-- retention rate
-- Tables: transactions, optionally users
WITH first_success AS (
    SELECT user_id,
        MIN(txn_ts) AS first_succ_txn
    FROM transactions
    WHERE status = 'success'
    GROUP BY user_id
),
cohorts AS (
    SELECT user_id,
        first_succ_txn,
        date(first_succ_txn, 'weekday 1', '-7 days') AS activation_week,
        datetime(first_succ_txn, '+7 days') AS d7_start,
        datetime(first_succ_txn, '+14 days') AS d14_end
    FROM first_success
),
retained AS (
    SELECT DISTINCT c.user_id,
        c.activation_week
    FROM cohorts c
        JOIN transactions t ON t.user_id = c.user_id
        AND t.status = 'success'
        AND t.txn_ts >= c.d7_start
        AND t.txn_ts < c.d14_end
)
SELECT c.activation_week,
    COUNT(*) AS activated_users,
    COUNT(r.user_id) AS retained_users,
    ROUND(1.0 * COUNT(r.user_id) / COUNT(*), 4) AS retention_rate
FROM cohorts c
    LEFT JOIN retained r ON r.user_id = c.user_id
GROUP BY c.activation_week
ORDER BY c.activation_week;
-- Find suspicious merchant segments
-- Question: Identify MCC + tier segments where both are true:
-- success rate is decent (>= 70%)
-- but refund rate is high (>= 20%)
-- Also compute average ticket size and total GMV.
-- Tables: transactions, refunds, merchants
-- Output: mcc, merchant_tier, success_rate, refund_rate, avg_amount_success, success_gmv
WITH refund_txns AS (
    SELECT DISTINCT txn_id
    FROM refunds
)
SELECT m.mcc,
    m.merchant_tier,
    COUNT(t.txn_id) AS total_txns,
    SUM(
        CASE
            WHEN t.status = 'success' THEN 1
            ELSE 0
        END
    ) AS succ_txns,
    COUNT(DISTINCT r.txn_id) AS refunded_txns,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN t.status = 'success' THEN 1
                ELSE 0
            END
        ) / NULLIF(COUNT(t.txn_id), 0),
        2
    ) AS success_rate_percent,
    ROUND(
        100.0 * COUNT(DISTINCT r.txn_id) / NULLIF(
            SUM(
                CASE
                    WHEN t.status = 'success' THEN 1
                    ELSE 0
                END
            ),
            0
        ),
        2
    ) AS refund_rate_percent,
    ROUND(
        AVG(
            CASE
                WHEN t.status = 'success' THEN t.amount
            END
        ),
        2
    ) AS avg_amount_success,
    ROUND(
        SUM(
            CASE
                WHEN t.status = 'success' THEN t.amount
                ELSE 0
            END
        ),
        2
    ) AS success_gmv
FROM transactions t
    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
    LEFT JOIN refund_txns r ON r.txn_id = t.txn_id
GROUP BY m.mcc,
    m.merchant_tier
HAVING SUM(
        CASE
            WHEN t.status = 'success' THEN 1
            ELSE 0
        END
    ) >= 30 -- min successful txns
    AND ROUND(
        100.0 * COUNT(DISTINCT r.txn_id) / NULLIF(
            SUM(
                CASE
                    WHEN t.status = 'success' THEN 1
                    ELSE 0
                END
            ),
            0
        ),
        2
    ) >= 20
ORDER BY refund_rate_percent DESC;
--- Learning cohorts
-- Monthly Signup Cohort Retention
WITH user_cohorts AS(
    SELECT user_id,
        date(signup_ts, 'start of month') AS cohort_month
    FROM users
)
SELECT uc.cohort_month,
    date(t.txn_ts, 'start of month') AS txn_month,
    COUNT(DISTINCT t.user_id) AS active_users
FROM user_cohorts uc
    JOIN transactions t ON uc.user_id = t.user_id
GROUP BY 1,
    2
ORDER BY 1,
    2;
WITH user_cohorts AS (
    SELECT user_id,
        date(signup_ts, 'start of month') AS cohort_month
    FROM users
),
txn_data AS (
    SELECT uc.cohort_month,
        t.user_id,
        date(t.txn_ts, 'start of month') AS txn_month,
        (
            strftime('%Y', t.txn_ts, 'start of month') - strftime('%Y', uc.cohort_month)
        ) * 12 + (
            strftime('%m', t.txn_ts, 'start of month') - strftime('%m', uc.cohort_month)
        ) AS month_number
    FROM user_cohorts uc
        JOIN transactions t ON uc.user_id = t.user_id
)
SELECT cohort_month,
    month_number,
    COUNT(DISTINCT user_id) AS active_users
FROM txn_data
GROUP BY 1,
    2
ORDER BY 1,
    2;

-- Behavioral Cohort (First 7 Day Intensity)
WITH first_week_txns AS (
    SELECT
        u.user_id,
        COUNT(*) AS txn_count_7d
    FROM users u
    JOIN transactions t
    ON u.user_id=t.user_id
    AND t.txn_ts <= datetime(u.signup_ts,'+7 days')
    GROUP BY 1
),
behavior_segment AS (
    SELECT 
        user_id,
        CASE WHEN txn_count_7d >=3 THEN 'High Intent'
        ELSE 'Low Intent'
        END AS segment
    FROM first_week_txns
)
SELECT * from behavior_segment