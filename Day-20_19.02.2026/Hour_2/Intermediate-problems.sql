-- Activation latency by channel (P50 / P90)
-- Question: For each acq_channel, compute:
-- P50 and P90 of time-to-activation = activation_ts - signup_ts in hours
-- Only include users who activated.
-- Output: acq_channel, activated_users, p50_hours, p90_hours
WITH hrs_per_acc_with_rn AS (
    SELECT user_id,
        acq_channel,
        (
            (unixepoch(activation_ts) - unixepoch(signup_ts)) / 3600.0
        ) AS hrs_to_Act,
        ROW_NUMBER() OVER (
            PARTITION BY acq_channel
            ORDER BY acq_channel,
                (unixepoch(activation_ts) - unixepoch(signup_ts))
        ) AS rn,
        COUNT(*) OVER (PARTITION BY acq_channel) AS n
    FROM users
    WHERE activation_ts IS NOT NULL
),
targets AS(
    SELECT *,
        CAST(ceil(0.5 * n) AS integer) AS p50_rn,
        CAST(ceil(0.9 * n) AS integer) AS p90_rn
    FROM hrs_per_acc_with_rn
)
SELECT acq_channel,
    COUNT(user_id) as activated_users,
    MAX(
        CASE
            WHEN rn = p50_rn THEN hrs_to_Act
        END
    ) AS p50_hours,
    MAX(
        CASE
            WHEN rn = p90_rn THEN hrs_to_Act
        END
    ) AS p90_hours
FROM targets
GROUP BY acq_channel;
-- Funnel leakage diagnosis: KYC done but no txn within 7 days
-- Question: For each signup week, compute:
-- users who did KYC within 7 days
-- among them, users with no successful txn within 7 days
-- leakage rate
-- Output: signup_week, kyc_7d, kyc_no_txn_7d, leakage_rate
WITH cohorts AS (
    SELECT user_id,
        -- Monday of signup week
        date(
            signup_ts,
            '-' || (
                (cast(strftime('%w', signup_ts) as integer) + 6) % 7
            ) || ' days'
        ) AS signup_week,
        signup_ts,
        kyc_ts
    FROM users
    WHERE signup_ts IS NOT NULL
),
kyc_7d AS (
    SELECT user_id,
        signup_week,
        signup_ts,
        kyc_ts
    FROM cohorts
    WHERE kyc_ts IS NOT NULL
        AND kyc_ts >= signup_ts
        AND kyc_ts < datetime(signup_ts, '+7 days')
),
kyc_no_txn_7d AS (
    SELECT k.user_id,
        k.signup_week
    FROM kyc_7d k
    WHERE NOT EXISTS (
            SELECT 1
            FROM transactions t
            WHERE t.user_id = k.user_id
                AND UPPER(t.status) = 'SUCCESS'
                AND t.txn_ts >= k.signup_ts
                AND t.txn_ts < datetime(k.signup_ts, '+7 days')
        )
)
SELECT k.signup_week,
    COUNT(*) AS kyc_7d,
    COALESCE(n.no_txn_7d, 0) AS kyc_no_txn_7d,
    ROUND(
        CASE
            WHEN COUNT(*) = 0 THEN 0
            ELSE 100.0 * COALESCE(n.no_txn_7d, 0) / COUNT(*)
        END,
        2
    ) AS leakage_rate
FROM kyc_7d k
    LEFT JOIN (
        SELECT signup_week,
            COUNT(*) AS no_txn_7d
        FROM kyc_no_txn_7d
        GROUP BY signup_week
    ) n ON n.signup_week = k.signup_week
GROUP BY k.signup_week
ORDER BY k.signup_week;
-- GMV concentration: top 1% users share of GMV
-- Question: Compute:
-- total successful GMV
-- GMV from top 1% users by successful GMV
-- share %
-- Output: total_gmv, top_1pct_gmv, top_1pct_share
WITH gmv_per_user AS (
    SELECT user_id,
        (SUM(amount)) as gmv,
        ROW_NUMBER() OVER (
            ORDER BY (SUM(amount)) DESC
        ) AS rn
    FROM transactions
    WHERE UPPER(status) = 'SUCCESS'
    GROUP BY 1
),
top_1pct_gmv AS (
    SELECT MAX(1, ceil(0.01 * COUNT(*))) as n
    FROM gmv_per_user
)
SELECT SUM(gmv) AS total_gmv,
    ROUND(
        SUM(
            CASE
                WHEN g.rn <= t.n THEN g.gmv
                ELSE 0
            END
        ),
        2
    ) as top_1pct_gmv,
    ROUND(
        (
            SUM(
                CASE
                    WHEN g.rn <= t.n THEN g.gmv
                    ELSE 0
                END
            ) / SUM(gmv)
        ) * 100.0,
        2
    ) AS top_1pct_share
FROM gmv_per_user g
    JOIN top_1pct_gmv t;
-- Power users” validation: 20% users produce what % of txns?
-- Question: Verify the Pareto claim in your generated data:
-- take users ordered by #successful txns
-- compute cumulative share
-- report % of successful txns produced by top 20% users
-- Output: top_20pct_users_txn_share
WITH succ_txns_per_user AS (
    SELECT user_id,
        COUNT(txn_id) AS succ_txns
    FROM transactions
    WHERE UPPER(status) = 'SUCCESS'
    GROUP BY 1
),
ranked AS (
    SELECT user_id,
        succ_txns,
        ROW_NUMBER() OVER (
            ORDER BY succ_txns DESC
        ) AS rn
    FROM succ_txns_per_user
),
top_20pct_users AS (
    SELECT CASE
            WHEN ceil(0.20 * COUNT(*)) < 1 THEN 1
            ELSE CAST(ceil(0.20 * COUNT(*)) AS integer)
        END AS n
    FROM succ_txns_per_user
)
SELECT ROUND(
        100.0 * SUM(
            CASE
                WHEN r.rn <= t.n THEN r.succ_txns
                ELSE 0
            END
        ) / SUM(r.succ_txns),
        2
    ) AS top_20pct_users_txn_share
FROM ranked r
    CROSS JOIN top_20pct_users t;
-- Campaign ROI: 30-day revenue vs spend (by campaign)
-- Question: For each campaign_id:
-- spend (sum marketing_spend)
-- acquired users (users with that campaign_id)
-- 30-day revenue from those users (sum rev_interchange + rev_mdr_share on successful txns within 30 days of signup)
-- ROI = revenue / spend
-- Output: campaign_id, spend, acquired_users, rev_30d, roi
WITH revenue_per_user_campaign AS(
    SELECT (t.user_id) as user_id,
        u.campaign_id as campaign_id,
        SUM((rev_interchange) +(rev_mdr_share)) AS revenue_per_user
    FROM users u
        RIGHT JOIN transactions t ON u.user_id = t.user_id
    WHERE t.status = 'success'
        AND t.txn_ts > datetime(signup_ts)
        AND t.txn_ts <= datetime(signup_ts, '+30 days')
    GROUP BY 1
),
total_revenue_per_campaign AS (
    SELECT campaign_id,
        SUM(revenue_per_user)
    FROM revenue_per_user_campaign
    GROUP BY 1
)
SELECT m.campaign_id,
    SUM(m.spend),
    COUNT(r.user_id) as acquired_users,
    SUM(r.revenue_per_user) as rev_30d
FROM revenue_per_user_campaign r
LEFT JOIN marketing_spend m ON m.campaign_id=r.campaign_id
GROUP BY 1;
--Not over