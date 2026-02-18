Topic : 
Basic ETL design (source → staging → warehouse → BI) 
Build simple ETL script (CSV → DB → transform → query)

1️⃣ Concept Clarity
A) Basic ETL design (Source → Staging → Warehouse → BI)
ETL exists for one reason: turn chaotic operational data into reliable decision data.

1) Source (OLTP / external feeds) - Sources change without warning. Columns get added, formats break, timestamps go weird.
app DB: users, sessions, transactions, refunds, loans

payment processor: settlement files

marketing: spend/campaign logs

risk: model scores, device fingerprinting

support: ticketing, disputes

2) Staging
This is where you dump data as-is.
    Minimal transformation

    Preserve everything for audit/debug

    Add ingestion metadata: ingested_at, source_file, batch_id

3) Warehouse (modeled, cleaned, consistent)
    standardized types (timestamps, currency)

    deduplication

    enrichment (join merchant metadata, geo, device)

    business logic: what counts as “successful txn,” “revenue,” “refund impact”

    SCD handling for dimensions

Connect to outcomes:

    revenue: net revenue after reversals/refunds

    risk: fraud/chargeback rates by segment over time

    retention: clean session/txn definitions per cohort

    unit economics: contribution margin by channel/segment

4) BI / Marts (serving layer)

Don’t let dashboards hit raw facts at 10B rows.
You build:

    aggregated tables (daily user activity, channel revenue daily)

    semantic definitions (DAU = distinct active users with session_start event)

    stable, performant datasets for BI tools

B) Incremental loads, idempotency, and late data

Incremental: load only new/changed records (by updated_at, event_ts, file date).
Idempotent: re-running the pipeline should not duplicate data.
Late-arriving data: refunds/chargebacks arrive days later; risk bands change later; settlement comes later.

C) Build simple ETL script (CSV → DB → transform → query) — what I expect you to be able to do

Minimal pipeline you should be able to build alone:

    Read CSV (transactions/users/refunds)

    Load to staging tables (raw types + ingestion metadata)

    Transform into warehouse tables (cleaned, typed, deduped)

    Build one mart table (daily channel metrics)

    Query it and validate with sanity checks


4️⃣ SQL
1) “Net revenue by channel daily (after settled refunds)”
2) “DAU and WAU trends with anomaly detection”
3) “Refund rate by merchant category and risk band (as-of txn time)”
4) “Cohort retention: users who transacted in week 0 → active in week 1/2/4”
5) “Unit economics: contribution margin by cohort/channel”


7️⃣ Interview Simulation
A) 3 realistic mid-level interview questions
