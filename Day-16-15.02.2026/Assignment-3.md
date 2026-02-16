# Assignment - 3


**A digital lending fintech reports:** 

Loan disbursements ↑ 18%

Default rate ↑ from 3% to 7%

Average credit score of new borrowers ↓ 40 points

Revenue ↑ 10%

### Insights : 
#### 1. What is the strategic mistake likely happening?
1. Loan disbursements increasing while Average credit score of new borrowers ↓ 40 points.
This means more people who may not repay the loan are getting disbursements.

#### 2. What business trade-off is management making?
Management is increasing loan disbursement volume discarding the fact that the new borrowers credit score is sub-par.
So management is trading off default rate to increase Loan Disbursement

#### 3. What 5 SQL breakdowns would you run?
1. Default rate by credit score bucket (e.g., 300–500, 500–650, 650+)

2. Approval rate by credit score bucket (before vs after policy change)

3. Cohort analysis of new borrowers (Month 1, Month 2 delinquency)

4. Loan size distribution shift (are smaller, riskier loans increasing?)

5. Net interest income vs expected credit loss (ECL) trend

6. Default rate by acquisition channel (organic vs marketing push)

#### 4. Would you recommend slowing growth? Why or why not?
I would not immediately slow growth. Instead, I would evaluate whether the increased yield from higher-risk borrowers compensates for increased expected loss. If risk-adjusted return remains positive, growth may still be justified. However, if expected credit loss exceeds incremental revenue, tightening underwriting thresholds is necessary.

#### 5. How would you quantify risk exposure?
Expected Credit Loss:
ECL = Probability of Default × Exposure at Default × Loss Given Default
If default rate increased from 3% → 7%:
On ₹100M portfolio:
Old expected loss = 3M
New expected loss = 7M

That’s +₹4M risk exposure.