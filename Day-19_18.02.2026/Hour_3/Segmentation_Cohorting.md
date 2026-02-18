1️⃣ First: What is Segmentation? What is Cohorting?

Segmentation = slicing your users/events into meaningful buckets to explain behavior differences.

Cohorting = grouping users by a shared starting event/time and tracking them over time.

2️⃣ Why This Matters - Segmentation turns numbers into decisions.

If you don’t segment, your metrics lie to you.

Example:

    Overall 30-day retention = 22%

Now segment:

    Paid acquisition users: 12%

    Organic users: 35%

    Referral users: 48%

3️⃣ Where Segmentation is Used in Fintech

A. Revenue Segmentation → To know where unit economics work.

Break revenue by:

    Acquisition channel

    Geography

    Risk bucket

    User tenure

    Merchant category

B. Risk Segmentation → Fraud rarely shows up evenly distributed.

Segment users by:

    First 3 transactions behavior

    KYC speed

    Device reuse

    Cashback abuse 

C. LTV Segmentation → Some cohorts are gold mines. Others burn cash.

Segment by:

    Signup month

    Credit score band

    Initial transaction amount

    Activation 


D. Behavioral Segmentation → Early signals predict retention.

Segment by:

**transactions in first 7 days**

    Average ticket size

    Time to activation

    Failed txn ratio


4️⃣ Cohort Types

1. Time-Based Cohort (Classic)
Group users by signup month.
Track:

    Retention

    Revenue

    txns

    Refund rate

2. Event-Based Cohort
Group by:

    First successful txn

    First loan disbursement

    First cashback redemption

Signup ≠ activation. In fintech, signup means nothing. Activation matters.

3. Behavior-Based Cohort
Group by:

    Users who did ≥3 txns in first 7 days

    Users who transacted ≥₹1000 first week

    Users who added card within 24h

4. Acquisition Cohort
Group by:

    Channel

    Campaign

    Adset

Track:

    CAC

    7d revenue

    30d retention

    Payback window

5. Risk Cohort
Group by:

    Risk_flag

    Device duplication

    Geo risk

5️⃣ When To Use Which Cohort?

1. Are you trying to explain time evolution?
→ Use time-based cohort.
2. Are you trying to compare groups?
→ Use segmentation.
3. Are you trying to predict future value early?
→ Use behavior-based cohort.
4. Are you allocating spend?
→ Use acquisition cohort.


7️⃣ Basic SQL: Monthly Signup Cohort Retention

WITH user_cohorts AS(
    SELECT 
        user_id,
        DATE_TRUNC('month',signup_ts) AS cohort_month
    FROM users
)
SELECT 
    uc.cohort_month,
    DATE_TRUNC('month',t.txn.ts) AS txn_month,
    COUNT(DISTINCT t.user_id) AS active_users
FROM user_cohorts uc
JOIN transactions t
ON uc.user_id=t.user_id
GROUP BY 1,2
ORDER BY 1,2;


-- 9️⃣Behavioral Cohort (First 7 Day Intensity)
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

11️⃣ Strategic Thinking Level

Which users deserve cashback?
Which channel to shut down?
Which early behavior predicts LTV?
Which risk band to tighten?
Which feature improved retention?


12️⃣ Interview-Level Advanced Thinking

How would you identify high LTV users early?
    Create behavior-based cohort from first 7–14 days
    Track long-term revenue by cohort
    Identify predictive signals
    Validate statistical significance
    Deploy segmentation into CRM


13️⃣

Q1:

You see 30-day retention dropped from 28% to 24%.
What cohort analysis would you run first?
Answer : 
Is this a specific signup cohort issue or a measurement issue?
1. Run time-based signup cohort retention
    If only recent cohorts are lower → something changed:

    Product

    Onboarding

    Traffic quality

    Risk filters

    Cashback structure

    If all cohorts shifted → maybe:

    Definition change

    Tracking bug

    Transaction classification change

2. THEN segment inside affected cohorts by:

    Acquisition channel

    App version

    Activation speed

    First txn amount



Q2:

You are asked: “Why is revenue flat despite user growth?”

What segmentation would you perform?

Revenue = Active Users × Txns per User × Avg Ticket × Take Rate

One of these is declining:

    Activation rate

    Engagement intensity

    Avg ticket size

    Take rate

    Or mix shifted to low-value users

1. Cohort by signup month -> See if new cohorts monetise worse.
2. Segment by acquisition channel -> Maybe scaling low-intent paid traffic.
3. Segment by user tenure -> Are new users contributing less than old users?
4. Segment by txn frequency bucket:-> Maybe growth is in low-frequency users.

Q3:

You notice referral users have 2x LTV.
What advanced segmentation would you run next?

    Why?

    Is it because they’re socially validated?

    Do they activate faster?

    Do they transact more frequently?

    Is fraud lower?

    Is retention higher?

Compare behavior metrics:
    Time to first txn

    Txns in first 7 days

    Failed txn rate

    Refund rate

    Risk_flag rate
Segment referral users by:
    Referrer quality (high-LTV referrers vs low)

    Reward amount

    Social graph depth
Check diminishing returns:
    Does LTV drop when referral volume scales?


Q4:

If you had to predict LTV in first 10 days, what 3 behavioral features would you test?

Users who transacted ≥₹1500 in first week
Users who added card within 24h
Users who did ≥5 txns in first 10 days

Txn velocity -> txns / days active
Avg ticket size
Revenue margin in first 10 days
Payment method diversity
Time to activation
Failed txn ratio
Refund ratio


### Strategic Segmentation Framework
strategic purposes for segmentation:
1️⃣ Diagnosis
2️⃣ Optimization
3️⃣ Prediction
4️⃣ Personalization
5️⃣ Risk Control


Advanced Cohort Types
Revenue Cohort -> Cohort by first transaction revenue bucket.

Survival Cohort -> Track probability of churn at each period.

Feature Adoption Cohort -> Users who used feature X in first week vs didn’t.

Pre/Post Intervention Cohort -> Users before cashback change vs after.


Q1:

If you cohort by signup month and see:

Jan cohort Month 1 retention = 40%
Feb cohort Month 1 retention = 32%
Mar cohort Month 1 retention = 24%

What are 5 hypotheses you generate immediately?

First classify the problem:
Retention drop is monotonic across recent cohorts →
This suggests something progressively worsening

1️⃣ Traffic Mix Shift -> Check acquisition channel mix per cohort.
    % Paid Search increased?

    % Incentivized traffic increased?

    % Low-quality geos increased?

2️⃣ Activation Rate Drop -> Month 0 activation rate per cohort.
Maybe users sign up but don’t activate.

3️⃣ Early Experience  -> Failed txn ratio in first 7 days per cohort.

    App crash rate increased?

    Failed txn ratio increased?

    Payment gateway failure?

4️⃣ Incentive Dilution ->     Cashback cost per activated user per cohort.

     reduced? Eligibility tightened?

5️⃣ Risk Policy Tightening -> % risk_flag users per cohort.
    Maybe you blocked borderline users who used to transact.

Q2:

You build a 7-day high-intent segment.
It shows 3x LTV.

How do you validate that this is causal and not correlation?

Step 1: Control for Acquisition Channel

Maybe high-intent users come mostly from referral.

So compare:
High-intent vs low-intent within same channel.

Step 2: Control for Risk Flag

High txn velocity might just mean high credit score users.

Control for risk bucket.

Step 3: Regression / Matching

Run:

LTV ~ txn_count_7d + acq_channel + geo + risk_flag

If txn_count_7d coefficient still strong → stronger causal signal.

Step 4: Intervention Test

Give targeted push notification to low-intent users:
Encourage 3rd transaction.

If LTV increases → causal.

That’s how product analytics validates causality.

Q3:

You segment by acquisition channel and see:

Paid Search users:

    High volume

    Low retention

    High CAC

    Low refund rate

Referral users:

    Low volume

    High retention

    Low CAC

    Slightly higher refund rate

Which channel do you scale?
How do you decide?


If:

Paid:
LTV = ₹800
CAC = ₹700
LTV/CAC = 1.14

Referral:
LTV = ₹1500
CAC = ₹300
LTV/CAC = 5

Then scaling referral is obvious.

But maybe referral is supply constrained.

Paid search may improve with targeting.