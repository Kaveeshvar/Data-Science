## Topics

Fact vs Dimension tables, granularity 
Slowly Changing Dimensions basics 
Indexing + query optimization concepts

1️⃣ Concept Clarity
A) Fact vs Dimension tables

Facts -> Numbers you measure ->Like sales, revenue, quantity.
 = events or measurable outcomes
Numeric metrics you aggregate: amount, revenue, fees, loss, count(txn)
Grain is sacred: 1 row per X (transaction / daily balance / loan EMI / card swipe / app session)

Dimensions -> Details that describe facts -> Like date, product, customer.
 = descriptive context used to slice facts.
user, merchant, product, geo, device, channel, risk_segment

**facts answer what happened, dimensions answer why/how/who/where**


B) Granularity (grain)
Granularity = what one row means.

C) Slowly Changing Dimensions (SCD)
Dimensions change: user address, KYC status, risk band, device, employer, plan tier.
SCD tells you how to store changes without destroying history.

SCD Type 1 (overwrite)
SCD Type 2 (history tracking)


D) Indexing + query optimization
Indexing is not “make query faster.” It’s:
reduce work by allowing the engine to find rows without scanning everything.
Optimization mindset:

    filter early

    aggregate early (if appropriate)

    avoid row explosion joins

    use the right grain table (don’t query txn-level for daily metrics unless needed)


4️⃣ SQL-Level Thinking
1) “What is contribution margin by acquisition channel for the last 30 days?”
2) “Retention curves: D1/D7/D30 transaction retention by signup cohort”
3) “Risk: Default rate by risk band at the time of loan disbursal (not current band)”
4) “Fraud spike detection: Compare hourly fraud rate vs trailing 7-day baseline”
5) “Performance: Why is this query slow and how do we fix it?”

7️⃣ Interview Simulation
A) 3 mid-level conceptual interview questions
1. You’re given fact_transactions and dim_user (SCD2). Explain exactly how you’d compute “revenue by risk band at time of transaction.” What columns do you require in dim_user to make this correct?
Define metric - > revenue ≠ amount
    Do we count only success?

    Do we net out reversed?

    Refund handling: initiated vs failed vs rejected should not all reduce revenue.

    Only settled/processed refunds should reduce realized revenue.

    Initiated can be a leading indicator, not a deduction.

Do an “as-of” join to SCD2
txn_ts must fall between user_dim’s effective_from/effective_to.
Columns required in dim_user_scd2
    user_id (natural key)

    risk_band

    effective_from

    effective_to (or is_current)

    surrogate key like user_sk (strongly recommended)

    updated_at (for debugging late-arriving changes)



2. A PM asks: “Why did DAU drop 12% WoW?” Your dataset has session facts at session-level grain and transaction facts at txn-level grain. Walk me through your debug plan and what joins you refuse to do.
metric definition
    DAU based on sessions? app_open? distinct user_id? timezone boundaries? bots filtered?

    Any tracking changes? missing events? app version rollout?

the drop
    New users vs returning users

    By platform (iOS/Android/Web)

    By app version

    By geo

    By acquisition channel / campaign

    By crash rate / latency / login 
    
where in the journey it broke
    install → signup → login → session start → key action → txn attempt → txn success

What joins do you refuse to do? 
* I refuse to join session fact to txn fact at raw level without pre-aggregating, because it creates many-to-many explosions -> 1 user has many sessions and many txns → cross product → fake correlations.
* I also refuse to join SCD2 user dim without an as-of condition.

3. When would you choose SCD Type 1 over Type 2 in a fintech environment? Give two examples where Type 2 is mandatory and why it impacts business decisions (risk/revenue/retention).
Type 1 good for:

    fixing typos in name, email casing, city spelling

    correcting wrongly assigned acquisition channel due to ETL bug (with audit)

Type 2 mandatory for:

    risk tier / underwriting segment (model monitoring, loss attribution)

    pricing plan / fee tier / credit limit band (unit economics, margin attribution)

    KYC status if it affects eligibility & compliance over time (regulatory audits)


B) 3 SQL query questions (based on these topics)

1. Grain sanity check:
Given fact_transactions(txn_id, user_id, txn_ts, amount, status) and dim_user_scd2(user_id, risk_band, effective_from, effective_to), write SQL to output:

* risk_band, txn_date, txn_count, total_amount
where risk_band is the user’s band as of txn_ts.

SELECT u.risk_band, t.txn_date, count(t.txn_id) as txn_count, sum(t.amount) as total_amount
FROM fact_transactions t
LEFT JOIN dim_user_scd2 u 
ON t.user_id = u.user_id
WHERE t.txn_ts >= u.effective_from AND t.txn_ts < u.effective_to


2. Row explosion trap:
You join fact_transactions to dim_acquisition (1 row per user) and to dim_user_scd2 (multiple rows per user). Your totals doubled. Write a query (or query structure) that proves where duplication happens and how you’d fix it.

prove duplication via counts (txn_id count vs distinct txn_id)
identify which join introduces multiplicity


3. Optimization thinking:
A query filtering last 7 days is slow: it uses WHERE DATE(txn_ts) >= DATE('now','-7 day'). Rewrite it to be index/partition-friendly and explain what index you’d propose on fact_transactions for common analytics patterns.

Partitioning = physically organizing table data by a column (usually date).
most common patterns are:
    per user over time (retention, frequency, LTV)

    time-range scans + status filtering
    So a reasonable composite index (engine-dependent) is:

    (user_id, txn_ts)
    and sometimes:

    (txn_ts, status) or (txn_date, status) if status is selective enough.