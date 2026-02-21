Topics : 
Fraud basics (velocity, anomaly, rule-based logic)
Chargeback and refund economics
Expected credit loss concept

1️⃣ Concept Clarity

A) Fraud basics: velocity, anomaly, rule-based logic (stop the bleeding without killing good users)

Fraud is not a “risk metric.” It’s a unit economics killer that hides inside growth. 

1) Velocity checks

Entities can be:

    user

    device

    card/payment instrument

    IP / subnet

    merchant

    beneficiary / bank account

    geo route 

Examples (payments/wallet):

    5 failed OTPs in 2 minutes

    8 payment attempts in 60 seconds

    3 new beneficiaries added + transfers within 10 mins

    10 cards tried on same device in 1 hour

Why velocity matters:
Fraud often has time-compression. Legit behavior has human pacing.


Business tie:

    Too strict → false positives → conversion drops → revenue down

    Too loose → losses explode + chargebacks + scheme penalties → margin down

2) Anomaly detection -> deviation from expected behavior

Patterns:

    unusual merchant category for user

    unusual hour-of-day

    sudden ticket size jump

    new device + high value + new beneficiary combo

Anomaly is typically:

    z-score / percentile threshold on amount vs user history

    peer group comparisons (similar users)

    sequence anomalies (event ordering oddities)

3) Rule-based logic

Rules are deterministic decisions:

    IF new_device AND amount>₹X AND 3 failures THEN block

    IF IP risk score high AND beneficiary_added_recently THEN step-up auth

But they rot:

    fraud adapts

    rules create holes

    rules create bias (over-block certain segments)


4) Fraud isn’t just “loss.” It’s also trust + retention

A legit user hit by a false positive = rage + churn.
A fraud user not blocked = chargeback + merchant complaints + regulator heat.

So you measure:

    fraud loss rate

    false positive rate

    step-up completion rate

    dispute rate

    net approval rate

B) Chargeback & refund economics
Refunds and chargebacks are not the same

Refunds

Refund = merchant/customer resolution, usually within product workflow.
Cost impact:

    reverse MDR/interchange (sometimes partial)

    processing fees may still apply

    cashback/incentives may not be clawed back

    operational cost (support)

Refund statuses matter:

    initiated vs processed vs failed vs rejected
    Only processed/settled should hit realized revenue.

Chargebacks

Chargeback = card network / bank dispute process (more expensive + more dangerous).

Costs include:

    chargeback amount reversal

    chargeback fees (scheme fees)

    representment costs

    higher risk monitoring programs if rates exceed thresholds

    potential merchant account consequences

Economics framing -> Net margin per txn can go from +₹3 to -₹80 when chargeback happens.


C) Expected Credit Loss (ECL) concept

ECL is how you convert “risk” into expected rupee loss, which impacts:

    pricing (APR)

    approval policy

    provisioning (finance)

    capital allocation

**ECL = PD × LGD × EAD**

PD: Probability of Default (within horizon)

LGD: Loss Given Default (after recoveries/collateral)

EAD: Exposure at Default (outstanding principal/credit used)


4️⃣ SQL-Level Thinking

1) Velocity fraud: “Users with >N txn attempts in 10 minutes, by device”

SQL: window functions over time buckets, counts per entity
Why: velocity is time-local; you need partition by entity + time range logic.

2) Fraud/anomaly: “New device + high amount + new merchant → approval rate vs baseline”

SQL: segmentation + conditional aggregation + joins to device first_seen
Why: anomaly signals are combinations; you test lift in fraud/chargebacks and drop in approvals.

3) Chargeback economics: “Net margin impact after chargebacks by merchant”

SQL: join txns ↔ chargebacks + compute contribution margin deltas
Why: chargebacks are rare but high cost; you need value-weighted analysis.

4) Refund health: “Refund processing time distribution and its impact on churn”

SQL: date diff + percentile metrics + cohorting
Why: refund delay is a trust metric; it predicts churn and support load.

5) ECL monitoring: “ECL by origination cohort and policy version”

SQL: cohort by origination month + join to ECL snapshots + rolling comparisons
Why: risk models drift; policy changes alter portfolio risk; you must attribute ECL moves to cohorts/policies.



A) 3 mid-level interview questions

You’re asked to reduce fraud loss rate by 20% without dropping payment success rate by more than 1%. What’s your approach? What levers do you pull (rules, step-up, velocity thresholds), and what guardrail metrics do you set?

Chargeback rate crossed a network threshold last month. Walk me through a diagnostic plan: segmentation, root causes, and which stakeholders you’d involve. How do you quantify the margin impact?

Explain ECL to a PM in 60 seconds, then tell me how you’d use PD/LGD/EAD to change pricing or approval policy for one segment.

B) 3 SQL query questions

Write SQL to flag velocity: users with ≥5 transaction attempts in any rolling 10-minute window, output user_id, window_start, attempt_count. (Explain your windowing approach.)

Write SQL to compute chargeback loss rate by merchant_id for last 90 days:

cb_count, cb_amount, cb_fees, cb_rate_by_count, cb_rate_by_amount
Join fact_transactions to fact_chargebacks. (Explain why you’d compute both count-rate and amount-rate.)

Write SQL to compute monthly ECL by origination cohort and policy_version:

sum(EAD), weighted_avg_PD, weighted_avg_LGD, total_ECL
using fact_loan_accounts + fact_ecl_snapshots + fact_credit_decisions. (Explain why weighting matters.)