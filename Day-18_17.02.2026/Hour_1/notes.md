## Topics

### LTV, CAC, Payback period, Contribution margin

### Risk-adjusted return (basic expected loss thinking)

### Segment thinking: cohort, channel, merchant category

1️⃣ Concept Clarity

### A) Contribution margin (CM)

Contribution Margin = Revenue − Variable Costs − Losses (if credit/risk applies)

Digital wallet example:
Revenue per txn = MDR share + interchange + float income

Lending example:
Revenue = interest + processing fees + late fees
Variable costs = cost of funds + collections + servicing
**CM (lending) ≈ (Interest + Fees) − (Cost of Funds + Servicing + Collections) − (Expected Loss)**

### B) CAC

CAC is cost to acquire an activated customer
For wallet, “activated” might mean first successful load + first spend.
For lending, it might mean first disbursed loan.

### C) LTV

LTV = Σ (Contribution Margin per period × retention probability) over time
For wallet:
LTV is driven by repeat usage, txn frequency, MDR/interchange, cross-sell, and cashback burn.
For lending:
LTV is driven by repeat borrowing, interest margin, and absolutely dominated by losses.

### D) Payback period

how long until CAC is recovered

Payback = CAC / Monthly (or Weekly) Contribution Margin
( payback is not “revenue payback.” It’s margin payback.)

In fintech, a decent growth team cares about:

- Wallet: payback often needs to be weeks to a few months (cashback heavy)
- Lending: payback might be months, but only if EL is controlled

### E) Risk-adjusted return

If lending exists, your “profit” is irrelevant unless it’s loss-adjusted.

Expected Loss (EL) = PD × LGD × EAD
Where,
PD (probability of default) - Chance that borrower will not repay.

LGD (loss given default) - How much money you lose if they don’t repay.

EAD (exposure at default) - Total loan amount at the time they default.

**Risk-adjusted margin ≈ Revenue − Costs − EL**

### F) Segment thinking

Cohort: users who activated in the same week/month
Use cohorts to see decay and quality drift.

Channel: performance marketing, organic, referrals, affiliates
Channels have different fraud risk, retention, cost structure.

Merchant category (MCC): groceries vs gaming vs jewelry vs fuel

## SQL

### Q1) What’s CAC by channel for activated users by week?

- define activation event
- cohort users by activation week
- join to marketing spend (week/channel)
- compute CAC = spend / activated_users

### Q2) 90-day LTV by acquisition cohort and channel (wallet)

- cohort users by activation week
- pull txns for first 90 days post-activation
- compute contribution margin per user
- aggregate median/percentiles (not just mean)

### Q3) Payback curve: cumulative margin vs CAC over time

- compute daily/weekly cumulative margin per cohort/channel
- compare to CAC baseline
- identify week where cumulative margin crosses CAC

### Q4) Risk-adjusted return by lending cohort (month 1 vs month 3 delinquency)

- cohort loans by disbursal month
- compute interest earned (or accrued) vs expected loss
- track dpd transitions over time

### Q5) Merchant category (MCC) mix shift and margin impact

- group txns by MCC and month
- compute % mix and CM per MCC
- attribute CM change to mix vs rate changes

### Common Mistakes

- Using revenue instead of contribution margin for LTV/payback.
- CAC = spend / signups. Wrong. Spend / Activated users
- One blended LTV number without cohorts/channels. That’s hiding variance.
- Ignoring cashback burn and processing fees.
- In lending: ignoring EL or treating defaults as “edge cases”.

### Mid-level thinking

- Defines activation rigorously and aligns CAC to it.
- Builds LTV as cohort × channel × segment, shows distribution and sensitivity.
- Forces risk-adjusted economics: margin after EL, not vanity interest income.
- Spots quality drift (new cohorts worse than old).
- Ties analysis to a decision: “Pause channel X” / “Cap cashback for MCC Y” / “Tighten risk policy for cohort Z”.

6️⃣ Quantification Drill

Scenario: You run a wallet + lending hybrid.

Wallet economics (per active user per month):

- 18 txns/month
- Avg txn amount ₹450
- Net revenue per txn (interchange + MDR share) = 0.35% of amount
- Processing cost per txn = ₹0.90
- Cashback = ₹12 per active user per month
- Support/fraud ops variable cost = ₹6 per active user per month

Acquisition:

- Channel A CAC (per activated user) = ₹220
- Monthly retention of actives = 70% (assume constant each month)

Lending cross-sell (only 20% of active users take a loan within 3 months):

- Avg principal ₹8,000
- Net interest+fees over 3 months = ₹680
- Cost of funds + servicing over 3 months = ₹260
- Expected Loss per loan (EL) = ₹310

Tasks (answer with numbers):

Compute monthly contribution margin per active user for wallet (₹).

Given:

- 18 txns/month
- Avg txn ₹450
- Revenue per txn = 0.35% of amount
- Processing cost per txn = ₹0.90
- Cashback per user per month = ₹12
- Ops cost per user per month = ₹6

Revenue per txn

0.35% × 450 = 0.0035 × 450 = ₹1.575

Monthly revenue per user

18 × 1.575 = ₹28.35

Monthly processing cost

18 × 0.90 = ₹16.20

Total variable cost

processing 16.20 + cashback 12 + ops 6 = ₹34.20

✅ Contribution margin per month

28.35 − 34.20 = ₹ -5.85

Compute 3-month wallet LTV using retention (month1 + month2 + month3).

Retention factor per month:

Month1: 1

Month2: 0.7

Month3: 0.7² = 0.49

Sum retention weights = 1 + 0.7 + 0.49 = 2.19

Monthly CM = -5.85

✅ 3-month wallet LTV

-5.85 × 2.19 = ₹ -12.8115 ≈ ₹ -12.81

Compute risk-adjusted contribution margin per loan (₹).

Given:

Interest+fees = ₹680

Cost of funds + servicing = ₹260

Expected loss = ₹310

✅ Risk-adjusted CM per loan

680 − 260 − 310 = ₹110

Compute expected lending LTV per activated user from cross-sell (20% take-rate).

Take-rate = 20% = 0.2

✅ Expected value per activated user

0.2 × 110 = ₹22

Combine (2)+(4): Does Channel A look payback-positive within 3 months vs CAC ₹220?

Total 3-month expected LTV per activated user:

Wallet (3m): -12.81

Lending EV (3m window implied): +22
✅ Total = ₹9.19

Compare to CAC ₹220:

9.19 << 220

✅ No payback. It’s massively negative.

**Even if lending works, wallet is subsidized; with these numbers we’d need either much higher txn frequency, better revenue rate, lower cashback, or higher lending take-rate / repeat borrowing to justify CAC.**

7️⃣ Interview Simulation

Your CAC improved 15% MoM, but payback got worse. Give 3 plausible reasons and how you’d validate each in data.

- Quality mix shifted (cheaper channel / weaker cohorts)
- Validate: payback by channel × activation cohort, check retention curve + CM/user shift.
- Unit economics degraded (cashback ↑, processing fees ↑, MCC mix shift)
- Validate: CM decomposition: txn frequency × amount × revenue rate − cost components.
- Risk/fraud/chargebacks rose (especially if incentives attract bad actors)
- Validate: fraud rate, dispute rate, loss per active, by channel × MCC × device fingerprint × geo.

In lending, a new acquisition channel has 2× conversion and 20% higher APR, but Risk says pause. What metrics do you pull to decide in 24 hours?

- Early delinquency: DPD0→DPD7→DPD30 roll rates (cohort curves)
- PD / EL proxy: default rate by weeks-on-book (MOB), compare vs baseline
- LGD / recovery: recovery rates, collections efficiency (if available)
- Risk-adjusted margin: (interest+fees − cost − EL) per disbursed loan
- Approval vs disbursal quality: conversion might be higher because underwriting is looser

**If early delinquency is spiking, APR doesn’t matter because EL explodes nonlinearly.**

Revenue ↑ can come from:

- txns/user ↑ (frequency)
- amount/txn ↑
- revenue rate ↑ (pricing/mix)

CM ↓ can happen if:

- cashback ↑ faster than revenue
- processing costs ↑
- fraud/disputes ↑
- MCC mix shifted to low-margin categories
- promo leakage / “bonus hunters” increased

**CM is revenue minus costs. So if revenue rose but CM fell, costs (or losses) rose faster than revenue.**
