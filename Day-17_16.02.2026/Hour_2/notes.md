# **Decompose Revenue: Price × Volume × Mix; Contribution margin Cohort analysis theory + retention curves**

## 1️⃣ Concept Clarity
### Decompose Revenue: Price × Volume × Mix (Fintech reality)
#### Price
    Payments: effective MDR(Merchant Discount Rate) / interchange / commission per txn

    Lending: effective yield = interest + fees − waivers − collections cost − credit losses

    Wallet: effective commission from billpay/recharge, float income (if applicable)
**Price is almost always a derived metric:**
    price per txn = net revenue / successful txns

    price per GMV = net revenue / GMV (take rate)

#### 2. Volume

**Volume is not just “#transactions”. It has layers:**

    Users (how many transacting)

    Frequency (tx/user)

    Ticket size (₹ per txn)

    Success rate (attempts vs success—fintech’s silent killer)

So “Volume ↑” could actually mean:

    more attempts but success flat (bad)

    success ↑ but low-margin use case (maybe bad)

    frequency ↑ among high-value cohorts (good)

#### 3. Mix
Mix means composition shifts:

    Merchant payments vs P2P

    Credit card vs UPI vs wallet balance

    Billpay vs recharge vs QR merchant

    Prime borrowers vs risky borrowers

    High-margin merchants vs low-margin merchants

    New users vs retained users

**Mix is where revenue can go up while “retention” stays flat.** Because the remaining users are richer / better / more monetizable.

#### 4. Contribution Margin
**Contribution = Net Revenue − Direct Variable Costs − Incentives − Fraud Loss − Credit Loss (if lending)**

### Cohort analysis theory + retention curves
#### 1. What a cohort actually is

A cohort is: **users who share the same starting event** (anchor), typically:

    signup date cohort (weak)

    first successful transaction cohort (better)

    KYC approved cohort (useful)

    first use-case cohort (P2P vs merchant) (very useful)

    acquisition channel cohort (paid vs organic) (for CAC/LTV)

Anchor must match the question:

    If you’re measuring product habit → anchor on first value event

    If you’re measuring onboarding friction → anchor on signup/KYC start

#### 2. Retention curve = “survival function” for value behavior

- **Given you activated, what % still transact after N weeks?**

        Steep early drop = bad activation quality or product not sticky

        Long flat tail = strong habit loop / utility

        Curve shift after a release = your change had real impact

##### **Transacting retention > login retention**

---
---

## 2️⃣ Business Application
Decomposition tells leadership **what to double down on**:

    If volume ↑ due to promos but contribution ↓ → stop

    If price ↑ due to reduced incentives → sustainable

    If mix ↑ due to high-margin use case adoption → invest

Cohorts tell you whether changes create **durable retention** or “one-time spikes”.

## 3️⃣ Data Modeling View
### Core tables
#### dim_user

    user_id (PK)

    signup_ts, channel, campaign_id

    KYC_status, KYC_approved_ts

    risk_segment, geo, device

#### fact_transactions (grain: txn_id)

    txn_id (PK), user_id (FK)

    txn_ts, txn_type (merchant/p2p/billpay)

    amount (GMV), status

    merchant_id, instrument_type (UPI/card/wallet)

    failure_reason_code

#### fact_payment_attempts (grain: attempt_id)

    attempt_id, txn_id, attempt_ts

    gateway, status, error_code

#### fact_revenue (grain: txn_id + revenue_type)

    txn_id, revenue_type (MDR/commission/interchange/fees)

    gross_revenue_amount

####  fact_costs_variable (grain: txn_id or user_event)

    txn_id: processing_cost

    user_id: KYC_cost, OTP_cost

####  fact_incentives (grain: incentive_id)

    user_id, txn_id, campaign_id

    incentive_amount

####  fact_fraud_losses / chargebacks

    txn_id, user_id

    confirmed_fraud_flag, loss_amount

####  fact_loans (if lending)

    loan_id, user_id

    disbursed_amount, interest, fees

    dpd, writeoff_flag, credit_loss

### Warehouse marts 

- mart_revenue_bridge_daily (price/volume/mix)

- mart_cohort_retention_weekly (cohort_week × week_n)

- mart_unit_economics_user_month (net contribution per user-month)

## 6️⃣ Quantification Drill

A wallet has two transaction types:

**Month 1**

Type A (Merchant QR): 10M tx, revenue/tx = ₹0.10, variable cost/tx = ₹0.04

Type B (Billpay): 2M tx, revenue/tx = ₹0.80, variable cost/tx = ₹0.20

**Month 2**

Total tx increased to 13M

Merchant QR tx = 12M

Billpay tx = 1M

Pricing unchanged

**Tasks**

Compute Month 1 total revenue + total contribution - 1.8M -> 0.6m + 1.2m

Compute Month 2 total revenue + total contribution - 1.32M -> 0.72m + 0.6m

Decompose revenue change into:

volume effect (total tx change) - 

mix effect (shift A vs B)

Answer: Did “growth” improve the business? Use contribution, not revenue.
No, Type B has better returns and in the second month, txns for type has gone down by 50% (From 2M to 1M)
Both type of transactions should improve, that said, type 2's performance is key to growth.

***You grew low-quality volume.**

## 7️⃣ Interview Simulation
#### 1.“Revenue is up 15% MoM but contribution is flat. Walk me through a price-volume-mix decomposition and the first 3 slices you’d check.”

Revenue = Price × Volume × Mix
Contribution = Revenue − Variable Cost − Incentives − Fraud

##### possibilities

    Price ↑ but cost per txn ↑ equally (e.g., higher MDR but also higher gateway cost)

    Volume ↑ but driven by low-margin segments

    Incentives ↑ eating margin

    Fraud/chargebacks ↑

    Credit losses ↑ (if lending)

##### First 3 slices:

Contribution per txn by txn_type

Incentive per txn trend

Fraud loss % by segment


#### 2.Design a cohort retention analysis to evaluate a change in onboarding (KYC flow). What’s your cohort anchor, success metric, and guardrails?
Cohort Anchor : signup date for A/B cohorts OR KYC start date   
Success metric : activation rate (first successful txn within 7 days)
Guardrails : 
    fraud rate

    KYC approval rate

    KYC cost per user

    payment success rate

    support ticket rate
#### 3. “How would you explain to a CEO why user retention is flat but revenue retention is rising? Give 3 hypotheses and how you’d validate each.”
user retention = % users active
revenue retention = revenue in week_n / revenue in week_0 cohort baseline

    ARPU ↑ among retained users
    → check revenue per retained user cohort-week

    Mix shift toward high-margin segments
    → check segment contribution % within retained cohort

    Incentive reduction
    → check contribution/txn and incentive/txn trend

    Lending attach rate ↑
    → check % retained users taking loans + loan yield