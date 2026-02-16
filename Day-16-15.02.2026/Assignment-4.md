# Assignment - 4

**A payments fintech reports:**

Customer acquisition cost (CAC) ↑ 35%

Revenue per user (ARPU) ↑ 5%

Retention ↓ from 82% to 75%

Marketing spend doubled

Do:
>Segmentation
Quantification
Trade-off
Action recommendation

---
### Insights
#### 1️⃣ What is the most dangerous metric here?
most dangerous metric is actually LTV/CAC deterioration.
Retention ↓
CAC ↑

#### 2️⃣ What financial risk does this create?
- LTV shrinking while CAC expands 
- Payback period increasing
- Negative contribution margin per user

#### 3️⃣ What SQL segmentation would you run first?
- Branchwise Retention percentage in Ascending order and their Average CAC per user
- Revenue % by Retained users vs Churned users
- CAC vs Total spending of the company.
- Marketing spend percentage per one bucket of revenue.(How much is spent per 10m,100m,500m in revenue)
- CAC per new user with the revenue they provide

Retention by acquisition channel (Google Ads vs organic vs referrals)
CAC by acquisition channel
LTV by cohort (Month 0, Month 1, Month 2 retention curve)
Marketing spend vs incremental revenue per cohort

#### 4️⃣ What action would you recommend — cut marketing, optimize retention, or something else?
I would not cut marketing blindly. I would segment acquisition channels and immediately pause high-CAC, low-retention channels. Simultaneously, invest in onboarding improvements to increase 30-day retention. The goal is restoring LTV/CAC > 3.


#### 5️⃣ Quantify how this affects LTV/CAC ratio (conceptually).
LTV ≈ ARPU × Average Lifetime

If retention drops from 82% to 75%, average lifetime reduces.
So if previously:

LTV = 100
CAC = 40
LTV/CAC = 2.5

Now:

LTV might drop to 90
CAC rises to 54

LTV/CAC = 1.67
