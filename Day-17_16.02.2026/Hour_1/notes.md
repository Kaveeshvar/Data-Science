## Metrics

### 1. North Star Framework
Pick one main metric that shows real growth. All teams focus on improving that one number.
###### It must satisfy 
- Value to user - they’d be sad if it disappeared
- Value to Business - drives revenue or lowers cost/risk
- Leading Indicator - moves before revenue does, but correlates strongly

###### Digital wallet NSM candidates
Bad: App opens, MAU, downloads
Mid: #Transactions (can be spammy micro-txs), GMV (can be unprofitable)
Good (often): “Weekly Active Transacting Users (WATU)” or “# users completing ≥N successful payments per week”

**Guardrails**
NSM: Weekly active transacting users
Guardrails: Fraud rate, payment success rate, cost per transaction, chargeback rate, KYC pass rate, NPA/Delinquency (if lending is attached), net revenue per user

---

### 2. Activation Framework
Helps users reach their “aha!” moment quickly. 
So they see value fast and want to stay.
- Activation is first moment of realized value.
- For a wallet, “value” is successful money movement with minimal friction

**A strong Activation definition (example):**
Activated user = completes KYC + links bank/UPI + makes first successful transaction within 7 days.

**Typical wallet activation funnel:**

Install → Sign up → Phone verified → KYC started → KYC approved → Payment instrument added (UPI/bank/card) → First attempt payment → First successful payment → Second transaction within 7 days (habit seed)

 - Fast activation reduces CAC payback time. Slow activation increases churn + support cost.

---
### 3. Retention Framework
Focus on keeping users coming back again and again.
Make sure they get value, so they don’t leave.

- Retention is not “came back.” Retention is repeat value consumption.

**In wallets, retention is use-case driven:**

    P2P transfers

    Merchant UPI payments

    Bill payments

    Recharge

    Subscription autopay

    Wallet balance storage / rewards

**Retention must be measured by behavioral cohorts and use-case cohorts.**

- First transaction date cohort (classic)

- AND by first use-case (P2P vs merchant vs bill pay)

- AND by segment (KYC vs non-KYC, salary users vs students, high-fraud geos, etc.)

**Key retention metrics for wallet:**

    D1/D7/D30 transacting retention (not login retention)

    Repeat rate: % users with ≥2 tx in 14 days

    Frequency: tx per active user per week

    Stickiness: WATU (Weekly Active Transacting Users)/MAU ( Monthly Active Users)

    Survival curve: probability user remains transacting after N weeks

    Reactivation: dormant → transacting again (and cost to reactivate)

---
### 4. Monetization Framework

Plan for how the product makes money.
Turn user value into revenue.

##### Wallet monetization is typically a mix:

1. Payments revenue

    MDR / interchange (cards), merchant fees (where applicable), partner commissions

2. Float / interest income

    depends on regulation + wallet balance structure

3. Lending (big one)

    interest + processing fees - cost of funds - credit losses

4. Cross-sell

    insurance, investments, subscriptions

5. Cost offsets

    reduced support calls, fewer failed tx, fewer fraud losses

- Monetization must be measured as contribution margin, not revenue.


**A practical fintech monetization stack:**

Gross Revenue 

minus Direct Costs (incentives/cashback, payment processing, SMS, KYC cost)

minus Fraud & Chargebacks

minus Credit Losses (ECL) if lending
 = Contribution Profit

**Contribution Profit = Gross Revenue - Direct Costs - Fraud & Chargebacks - Credit Losses**

---
---

## Business Application

CEO: cares about NSM + contribution margin + risk. Wants scalable growth without landmines.

PM (Wallet/Payments): cares about activation funnel, success rates, retention loops.

Growth team: cares about CAC → activation rate → payback period.

Risk Head / Fraud: cares about fraud rate, KYC quality, chargebacks, delinquency, and policy impacts on activation.

Finance: cares about unit economics, incentives burn, take rate, and forecasting.

Ops/Support: cares about failure reasons, reversals, dispute rates.

    Activation is where your CAC gets justified or wasted.

    Retention is where LTV becomes real (not PowerPoint).

    Monetization is where you prove you’re not running a charity.

    NSM aligns teams. Without it, every function optimizes their local metric and the company becomes a KPI civil war.

---

---
## Data Modeling View
#### Core tables (typical)

1) users (1 row per user)

        user_id (PK)

        signup_ts, channel, referral_code

        KYC_status, KYC_approved_ts

        risk_segment, city/state, device_id

2) accounts / wallets

        account_id (PK), user_id (FK)

        wallet_status, balance, created_ts

        linked_bank_flag, upi_handle

3) transactions (fact table; this is the heartbeat)

        txn_id (PK)

        user_id (FK)

        txn_ts

        txn_type (P2P, merchant, billpay, recharge, add_money, withdraw)

        amount

        status (success/fail/pending/reversed)

        failure_reason_code

        merchant_id (nullable)

        counterparty_user_id (nullable)

4) payment_attempts (granular reliability)

        attempt_id (PK), txn_id (FK)

        attempt_ts, gateway, status, error_code
        This is how you analyze success rate properly.

5) incentives

        incentive_id, user_id, txn_id

        incentive_amount, campaign_id, issued_ts

6) revenue_ledger

        txn_id, user_id

        revenue_type (MDR, commission, interchange, late_fee)

        revenue_amount, booked_ts

7) fraud_cases / chargebacks

        case_id, user_id, txn_id

        fraud_flag, confirmed_fraud, loss_amount, reported_ts

8) lending_loans (if lending exists)

        loan_id, user_id

        disbursed_amount, disbursed_ts

        interest_rate, tenure

        dpd (days past due), status

        writeoff_flag, loss_amount

---
---

## SQL-Level Thinking
#### 5 analytical questions → what SQL logic is used

##### 1. What’s true activation rate in 7 days, and where is drop-off biggest?
 - Logic: funnel steps using event timestamps and left joins + min(ts) per user.
 - Why: you need ordered milestones (KYC approved before first successful txn).
 - Tools: conditional aggregation, MIN(CASE WHEN...), funnel stage flags.

 ##### 2.  Cohort retention by first successful txn week, split by first use-case
 - Logic: define cohort = first_success_txn_week using window functions (MIN(txn_ts) OVER (PARTITION BY user_id)), then compute week_n activity using date_diff buckets.
 - Why: retention is about time since value, not calendar time.
- Tools: window functions, cohort tables, grouping by cohort_week, week_number. 
#### 3. Did payment success rate drop last week and how did it impact retention?
- Logic: compute success rate by day/gateway using attempts, then correlate with next-week transacting retention by cohort.
- Why: reliability issues create churn lag.
- Tools: time series aggregation + joining success-rate metrics to cohort outcomes.
#### 4. Unit economics by segment: contribution profit per transacting user
- Logic: join transactions with revenue_ledger and incentives and fraud_loss; aggregate per user-week.
- Why: growth is fake if contribution < 0.
- Tools: multi-fact joins, careful deduping (revenue types), SUM() with grain control

#### 5. Which campaigns create retained users vs one-time incentive hunters?
- Logic: campaign exposure → first txn → 30-day retained flag; compare distributions and incremental lift using matched cohorts.

- Why: cashback can inflate activation but destroy unit economics.

- Tools: segmentation, cohort flags, propensity-ish matching (at least controlling for channel/segment).

---
---
## Common Mistakes
1. NSM = MAU : **NSM tied to value + profit + risk, with guardrails.**
2. Activation = signup/KYC : **activation = first successful value event + time-to-value.**
3. Retention measured on logins : **retention = transacting retention and use-case retention.**
4. Ignore reliability : **payment failures are leading indicator of churn; analyze attempts + error codes.**
5. No grain discipline : **you protect the metric from duplication (attempt vs txn vs revenue rows).**
6. Monetization = revenue : **monetization = contribution margin, net of incentives, processing, fraud, credit losses.**

---
---
## Quantification Drill
A wallet has **1,000,000 new signups/month.**

    60% complete KYC

    Of KYC-approved users, 50% add UPI

    Of UPI-added users, 40% make a first successful transaction within 7 days

    Of activated users, 30-day transacting retention is 25%

    Retained users do 12 transactions/month on average

    Net contribution per transaction = ₹0.35 (after incentives + processing)

**Tasks (calculate):**

How many activated users? - 120000

How many retained at 30d? - 30000

Monthly contribution profit from retained users? - ₹126000

If you improve UPI add rate from 50% → 60%, what is the incremental monthly contribution profit (assuming everything else same)? Total contribution = 1,51,200 | increment = 25,200

**Is improving UPI add rate profitable if it increases KYC cost, fraud, drop-offs, or incentives burn?**

---
---

## Interview Simulation 

Design a North Star metric for a wallet that is adding lending next quarter. What are your guardrails and why?
NSM = WATU with ≥2 successful tx
Guardrails = Fraud rate / chargeback rate, Take rate / net contribution per user, Payment success rate by instrument + gateway

Activation dropped 8% WoW. What are your first 5 slices, what tables do you hit, and what would you suspect in fintech specifically?

        Definition check: which step dropped? (KYC? UPI add? first attempt? success?)

        Decompose: volume vs conversion vs timing (time-to-activation shifted?)

        Instrument: payment attempts, gateway, PSP, error code distribution

        Segment: new channel mix, geo mix, device/OS, risk segments

        Change log: product release, KYC vendor, payment partner incident, policy changes

**Tables** :
        users (signup cohorts, channel)

        kyc_events or fields in users (start/approved timestamps)

        payment_instruments (UPI added)

        payment_attempts (error codes, gateway)

        transactions (success status, failure reasons)

**Fintech-specific suspects** : 

        PSP/gateway degradation

        UPI downtime / bank partner issue

        KYC vendor SLA hit

        new fraud rules blocking legit users

        Android build bug breaking UPI intent flow

        incentive removal impacting “first tx”

Retention is flat but revenue is up. Give 3 explanations that are not stupid, and how you’d validate each in data.

    Monetization per user increased (pricing, MDR, commissions, lending attach)

    Mix shift: same retention, but more high-value segments retained

    Frequency / ticket size up among retained users, even if count unchanged

    Incentives reduced so net revenue rises while behavior unchanged

    Revenue recognition timing (accounting shift)

**Validation**:

    revenue per transacting user

    contribution per txn

    segment-level revenue decomposition

    attach rates (bill pay, credit, insurance)

    ARPU vs #active users vs frequency

---
---
## Upgrade Task (30–45 min, applied) 
You’re going to build a mini-metrics spec + SQL plan like a real analytics lead.

**Deliverables (in one doc):**
1. Define for a wallet:

        NSM - # Weekly Active Users with ≥2 successful transactions

        Activation definition (with time window) - Activated user = completes KYC + links bank/UPI + makes first successful transaction within 7 days + Second Successful Transaction within 15 days.

        Retention definition (D7 + D30) - % of activated users who transact at least once in D7 / D30

        Monetization metric (contribution margin) - Contribution Profit = Gross Revenue - Direct Costs - Fraud & Chargebacks - Credit Losses


2. Write a table-level plan: which tables + grain + join keys for each metric.

**NSM (WATU ≥2 successful tx):**

* Primary: `transactions` (grain: txn_id)
* Keys: user_id
* Filters: status='success', txn_ts in week
* Output grain: user_id-week

**Activation (first success within 7d of signup):**

* `users` (grain: user_id)
* `transactions` (grain: txn_id)
* Key: user_id
* Logic: min(success_txn_ts) <= signup_ts + 7 days

**Retention D30 (transacting retention of activated cohort):**

* Cohort table derived from activation: (user_id, activated_date)
* Join to `transactions` for activity window: [activated_date+30] etc.
* Output: cohort_date, retained_flag

**Monetization (contribution per retained user-month):**

* `transactions` join `revenue_ledger` join `incentives` join `fraud_cases`
* Grain discipline: aggregate each to txn_id first, then join
* Output grain: user_id-month

**Guardrail 1: payment success rate**

* `payment_attempts` (grain: attempt_id), grouped by day/gateway
* metric = success_attempts / total_attempts

**Guardrail 2: fraud loss rate**

* `fraud_cases` joined to txns
* metric = confirmed_fraud_loss / GMV OR / successful_txns


3. Write pseudo-SQL (not perfect syntax) for:

        Activation rate in 7 days

        D30 transacting retention cohort table

        Contribution margin per retained user-month

Constraint:
- Include at least 2 guardrails (fraud + success rate) and explain how they’re computed.