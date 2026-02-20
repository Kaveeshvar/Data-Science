Topics : 
Retention levers (onboarding, habit loop, incentives) 
Growth loops vs linear growth 
Funnel drop-off diagnostics

1️⃣ Concept Clarity

A) Retention levers

Retention is not a KPI. Retention is a system. And in fintech, retention is chained to trust + frequency + value capture.

1) Onboarding


Onboarding is the phase where you convert intent into habit potential.
Your only job: get the user to the first meaningful value moment fast.
    Wallet: first successful add money + first payment

    Lending: first eligibility check + first disbursal (or at least approval)

    Credit card: first successful transaction + statement clarity

Measure : 
    Time-to-first-value (TTFV)

    Drop-offs by step

    Error reasons (KYC fail, bank link fail, OTP fail)

    “First success” rate (first successful txn within 24h/7d)

**Better onboarding → higher activation → higher LTV → higher CAC payback → higher growth budget.**

Bad onboarding → lower activation → you think you need “more marketing” → you burn money.

2) Habit loop (Trigger → Action → Reward → Investment)

Habit loop means the product becomes part of life.
Fintech habit is usually weekly or daily when:

    bills

    salary

    spending

    savings goals

    credit repayment

Examples : 
Trigger: salary credited → Action: auto-sweep to savings → Reward: progress / interest / streak → Investment: setting goals, linking accounts

Trigger: daily spend → Action: scan-pay → Reward: fast checkout + cashback → Investment: saved merchants, UPI autopay, contacts

In Unit Economics : 
**Habit loop raises frequency**
Revenue = users × frequency × margin-per-txn
If frequency goes up, even flat margins print money.

3) Incentives
Incentives are not retention. Incentives are behavior shaping.

Good incentives:

    push users toward sticky behavior (autopay setup, first merchant repeat, recurring deposits)
    Bad incentives:

    “cashback for anything” → attracts deal hunters, increases cost, tanks margin

In lending:

    incentives can increase repayment discipline (on-time repayment rewards)
    But risk tradeoff: -> incentives can also attract risky users gaming rewards → fraud / losses rise.

Incentives must be evaluated on:

    incremental retention lift

    incremental margin impact

    incremental risk impact
        Not “redemptions.”



B) Growth loops vs linear growth
Linear growth (paid / outbound / one-shot)

    You push users in: ads, sales, partnerships

    Growth stops when spend stops

    CAC is the limiting factor

Growth loops (compounding systems)

    A loop is when output of the system becomes input again.

Fintech loops examples:

    Referral loop: happy users invite friends → friends activate → more inviters

    Merchant acceptance loop: more users → merchants see demand → accept → better coverage → more users

    Data/risk loop (lending): more borrowers → more repayment data → better underwriting → better approvals/lower losses → better pricing → more borrowers

    Liquidity/usage loop (wallet): more transactions → more stored value / trust → more transactions

Head-of-Analytics perspective:

    A loop has a cycle time, conversion at each step, and decay

    If cycle time is long or conversion is weak, loop doesn’t compound—it limps.


C) Funnel drop-off diagnostics

A funnel drop is : 
    Intent mismatch (wrong acquisition)

    Friction (OTP failure, KYC drop, bank link error, app crash)

    Trust breakdown (failed txn, stuck refund, support delays)

    Value not delivered fast enough

    Economics changed (fees increased, incentive removed)

    External shock (bank downtime, RBI rule changes, competitor promo)

**Your job: isolate which step, which segment, since when, why.**

You don’t “analyze drop.” You:

    locate (step)

    scope (segment)

    timestamp (change point)

    attribute (root cause candidate)

    quantify impact (lost actives, lost margin)


4️⃣ SQL-Level Thinking

1) “D1/D7 retention by acquisition channel (activated users only)”

Logic: cohorting + segmentation + left joins
Why: retention without activation filtering is misleading. You must define cohort anchor (signup_date vs first_value_date).

2) “Onboarding funnel: signup → OTP → KYC → bank link → first success”

Logic: event funnels + step completion flags + drop-off computation
Why: funnel requires ordering and step attribution, not just counts. You need earliest timestamp per step per user.

3) “Habit loop: weekly repeat rate after first success”

Logic: window functions (lag/lead) + time-bucketing
Why: habit isn’t DAU. It’s repeat behavior cadence. You measure inter-transaction gaps and repeat within target window.

4) “Impact of incentive removal on retention and margin”

Logic: diff-in-diff-ish segmentation + pre/post comparison + margin calculation
Why: incentives change both behavior and cost. If retention up but margin down, you might be paying for fake growth.

5) “Growth loop health: referral sent → accepted → activated → rewarded, with cycle time”

Logic: multi-step conversion + median time-to-convert + cohort tracking
Why: loops compound only if cycle time is short and conversion is high enough.


A) 3 mid-level interview questions

Your wallet product has stable signups but D7 retention dropped from 22% → 16% in two weeks. Outline your diagnosis plan: what funnels, what segments, what “change point” checks, and how you’ll quantify business impact.

Given that wallet has stable signups but D7 retenion dropped from 22% to 16%. 
1. Create activation->D7 funnel cohort with activation_ts. Check for delays/fails in any steps
2. Segment users based on acq_channel to understand users who onboarded with wrong intention/false hopes (Intent mismatch)
3. I predict this won't be a change in unit economics change because D7 is too early for that.
4. Cohort with merchant category to find out fail rates per merchant.


A) 3 mid-level interview questions

Design one growth loop for a lending product that compounds responsibly. Define the loop steps, success metrics at each step, cycle time, and the key risk/fraud failure modes you’d monitor.

Loop Steps (with measurable conversion at each step)

    Acquisition → application started

    Application → KYC complete

    KYC → approval

    Approval → disbursal

    Disbursal → first repayment due

    Due → on-time repayment

    On-time repayment → limit increase / better APR

    Better terms → repeat borrowing + referrals → back to step 1


Success metrics per step (one metric per step)

    Step 3: approval rate within risk band

    Step 4: median time-to-disbursal

    Step 6: DPD0 / DPD7 / roll-rate into delinquency

    Step 7: incremental CL increase accuracy + margin impact

    Loop-level: repeat borrow rate within 90 days, CAC payback

Cycle time -> define one: 30D / 60D / 90D loop (most lending loops are 60–120 days)


Risk/fraud failure modes you monitor (concrete)

    Synthetic identity / mule accounts passing KYC

    First-party fraud (intentional default)

    Collusion rings / device farms

    Repayment masking: partial payments to avoid delinquency tags

    Adverse selection: incentive attracts high-risk applicants

    Model drift: underwriting performance degrades by cohort/week



Incentives increased repeat transactions by 8%, but contribution margin dropped by 15%. Walk me through how you decide whether to keep, modify, or kill the incentive. What metrics are non-negotiable?

First Thought - Kill.
But check if there is incremental retention, reduction of risk impact.
And then modify if metrics align. 
I'd say no keep.




CM drop is non-negotiable EVEN IF retention rises, because you can buy retention with cash until you go bankrupt.

Incentives are a trade: short-term volume vs long-term profitability + risk + habit formation.

You only keep incentives if incremental LTV – incremental cost – incremental loss is positive.

You decide based on:

    Incrementality: would those repeat txns have happened anyway? (control vs treated)

    Margin decomposition: is CM down due to cashback cost, higher processing fees, lower MDR mix, higher refunds, higher fraud?

    Quality shift: are you attracting deal hunters / fraud?

    Persistence: does behavior persist after incentive removal? (habit vs bribery)

Non-negotiable metrics
    Incremental contribution margin per user (or per txn)

    LTV:CAC (or payback period if you can estimate)

    Fraud rate / chargeback rate / first-payment default (lending) / refund rate (wallet)

    Repeat rate after incentive ends (retention durability)

    Mix shift: merchant categories / payment rails / ticket size


B) 3 SQL query questions
Build a D7 retention query: users who had their first successful transaction (activation) on day 0, and whether they had any session on day 7. Segment by acq_channel. (Explain your cohort anchor and why.)

WITH retention AS (
    SELECT u.activation_ts, t.txn_ts, u.acq_channel
    FROM users & txn ON ...
    AND txn_status = 'success'
    AND txn_ts <= date(activation_tx,'+7 days')
 ),weekly_cohorts(...)
SELECT Week,acq_channel,D7 retention count, rate
FROM retention,weekly cohorts
Group by 1
order by 1

Funnel query: for each signup_date and app_version, compute step completion rates for: signup_submit → otp_success → kyc_approved → first_txn_success. Use event timestamps to ensure ordering. (Explain how you prevent double-counting users with repeated events.)

WITH per_user AS (
  SELECT
    e.user_id,
    date(MIN(CASE WHEN e.event_name = 'signup_submit' THEN e.event_ts END)) AS signup_date,
    -- pick app_version at signup_submit time (earliest signup_submit)
    (
      SELECT e2.app_version
      FROM events e2
      WHERE e2.user_id = e.user_id
        AND e2.event_name = 'signup_submit'
      ORDER BY e2.event_ts
      LIMIT 1
    ) AS app_version,

    MIN(CASE WHEN e.event_name = 'signup_submit'     THEN e.event_ts END) AS t_signup,
    MIN(CASE WHEN e.event_name = 'otp_success'       THEN e.event_ts END) AS t_otp,
    MIN(CASE WHEN e.event_name = 'kyc_approved'      THEN e.event_ts END) AS t_kyc,
    MIN(CASE WHEN e.event_name = 'first_txn_success' THEN e.event_ts END) AS t_txn
  FROM events e
  GROUP BY e.user_id
),
ordered AS (
  SELECT
    signup_date,
    app_version,
    user_id,

    t_signup,

    CASE
      WHEN t_otp IS NOT NULL AND t_signup IS NOT NULL AND t_otp >= t_signup
      THEN t_otp ELSE NULL
    END AS t_otp_ok,

    CASE
      WHEN t_kyc IS NOT NULL AND t_otp IS NOT NULL AND t_signup IS NOT NULL
       AND t_otp >= t_signup
       AND t_kyc >= t_otp
      THEN t_kyc ELSE NULL
    END AS t_kyc_ok,

    CASE
      WHEN t_txn IS NOT NULL AND t_kyc IS NOT NULL AND t_otp IS NOT NULL AND t_signup IS NOT NULL
       AND t_otp >= t_signup
       AND t_kyc >= t_otp
       AND t_txn >= t_kyc
      THEN t_txn ELSE NULL
    END AS t_txn_ok
  FROM per_user
  WHERE signup_date IS NOT NULL
)
SELECT
  signup_date,
  app_version,

  COUNT(*) AS signup_submit_users,

  SUM(CASE WHEN t_otp_ok IS NOT NULL THEN 1 ELSE 0 END) AS otp_success_users,
  ROUND(100.0 * SUM(CASE WHEN t_otp_ok IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS otp_success_rate_pct,

  SUM(CASE WHEN t_kyc_ok IS NOT NULL THEN 1 ELSE 0 END) AS kyc_approved_users,
  ROUND(100.0 * SUM(CASE WHEN t_kyc_ok IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS kyc_approved_rate_pct,

  SUM(CASE WHEN t_txn_ok IS NOT NULL THEN 1 ELSE 0 END) AS first_txn_success_users,
  ROUND(100.0 * SUM(CASE WHEN t_txn_ok IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS first_txn_success_rate_pct

FROM ordered
GROUP BY signup_date, app_version
ORDER BY signup_date, app_version;


Growth loop query (referrals): compute weekly conversion rates for referral_sent → accepted → activated → reward_issued, plus median time from sent → activated. Segment by inviter’s tenure band (0–7 days, 8–30, 31+). (Explain why tenure segmentation matters.)

WITH base AS (
  SELECT
    r.referral_id,
    r.inviter_user_id,
    r.invitee_user_id,
    r.referral_sent_ts AS t_sent,
    r.referral_accepted_ts AS t_accepted,
    r.reward_issued_ts AS t_reward,
    u_inv.signup_ts AS inviter_signup_ts,
    u_inv.activation_ts AS inviter_activation_ts,
    u_in.signup_ts AS invitee_signup_ts,
    u_in.activation_ts AS invitee_activation_ts,

    -- Monday of sent week
    date(
      r.referral_sent_ts,
      '-' || ((cast(strftime('%w', r.referral_sent_ts) as integer) + 6) % 7) || ' days'
    ) AS sent_week
  FROM referrals r
  LEFT JOIN users u_inv ON u_inv.user_id = r.inviter_user_id
  LEFT JOIN users u_in  ON u_in.user_id  = r.invitee_user_id
  WHERE r.referral_sent_ts IS NOT NULL
),
segmented AS (
  SELECT
    *,
    -- tenure of inviter at time of sent (days since inviter signup)
    CAST((julianday(t_sent) - julianday(inviter_signup_ts)) AS integer) AS inviter_tenure_days,
    CASE
      WHEN inviter_signup_ts IS NULL THEN 'unknown'
      WHEN (julianday(t_sent) - julianday(inviter_signup_ts)) <= 7  THEN '0-7'
      WHEN (julianday(t_sent) - julianday(inviter_signup_ts)) <= 30 THEN '8-30'
      ELSE '31+'
    END AS inviter_tenure_band,

    -- enforce ordering for conversions
    CASE
      WHEN t_accepted IS NOT NULL AND t_accepted >= t_sent THEN 1 ELSE 0
    END AS accepted_ok,

    CASE
      WHEN invitee_activation_ts IS NOT NULL
       AND t_accepted IS NOT NULL
       AND t_accepted >= t_sent
       AND invitee_activation_ts >= t_accepted
      THEN 1 ELSE 0
    END AS activated_ok,

    CASE
      WHEN t_reward IS NOT NULL
       AND invitee_activation_ts IS NOT NULL
       AND t_reward >= invitee_activation_ts
      THEN 1 ELSE 0
    END AS reward_ok,

    CASE
      WHEN invitee_activation_ts IS NOT NULL AND invitee_activation_ts >= t_sent
      THEN (julianday(invitee_activation_ts) - julianday(t_sent)) * 86400.0
      ELSE NULL
    END AS sec_sent_to_activated
  FROM base
),
agg AS (
  SELECT
    sent_week,
    inviter_tenure_band,

    COUNT(*) AS sent,
    SUM(accepted_ok) AS accepted,
    SUM(activated_ok) AS activated,
    SUM(reward_ok) AS reward_issued
  FROM segmented
  GROUP BY sent_week, inviter_tenure_band
),
median_prep AS (
  SELECT
    sent_week,
    inviter_tenure_band,
    sec_sent_to_activated,
    ROW_NUMBER() OVER (
      PARTITION BY sent_week, inviter_tenure_band
      ORDER BY sec_sent_to_activated
    ) AS rn,
    COUNT(sec_sent_to_activated) OVER (
      PARTITION BY sent_week, inviter_tenure_band
    ) AS cnt
  FROM segmented
  WHERE sec_sent_to_activated IS NOT NULL
),
median_calc AS (
  SELECT
    sent_week,
    inviter_tenure_band,
    AVG(sec_sent_to_activated) AS median_sec_sent_to_activated
  FROM median_prep
  WHERE rn IN ( (cnt + 1)/2, (cnt + 2)/2 )   -- works for odd/even
  GROUP BY sent_week, inviter_tenure_band
)
SELECT
  a.sent_week,
  a.inviter_tenure_band,

  a.sent,
  a.accepted,
  ROUND(100.0 * a.accepted / a.sent, 2) AS sent_to_accepted_pct,

  a.activated,
  ROUND(100.0 * a.activated / a.sent, 2) AS sent_to_activated_pct,
  ROUND(
    CASE WHEN a.accepted = 0 THEN 0 ELSE 100.0 * a.activated / a.accepted END,
    2
  ) AS accepted_to_activated_pct,

  a.reward_issued,
  ROUND(
    CASE WHEN a.activated = 0 THEN 0 ELSE 100.0 * a.reward_issued / a.activated END,
    2
  ) AS activated_to_reward_pct,

  ROUND(COALESCE(m.median_sec_sent_to_activated, 0) / 3600.0, 2) AS median_hours_sent_to_activated
FROM agg a
LEFT JOIN median_calc m
  ON a.sent_week = m.sent_week
 AND a.inviter_tenure_band = m.inviter_tenure_band
ORDER BY a.sent_week, a.inviter_tenure_band;
