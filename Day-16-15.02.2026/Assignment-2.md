# Assignment - 2

A fintech company’s:

- Transaction volume ↑ 20%

- Revenue ↑ 12%

- Profit margin ↓ from 22% to 16%

- Refund rate ↑ from 2% to 6%
---
## Insights : 
###  1. What is the immediate red flag?
- Refund rate tripled (2% → 6%) That's a 200% increase.
- Refunds directly:
    Reduce revenue
    Increase operational cost
    Potentially signal fraud
    Damage merchant trust


###  2. What 5 precise metrics would you query first?
- Refund amount / total transaction value
- Refund count by merchant category
- Gross transaction value (GTV) vs Net revenue
- Variable cost per transaction trend
- Discount / cashback usage rate
- Chargeback rate
- Fraud flag rate

###  3. What SQL-level breakdown would you run?
- Refund rate by merchant_category, day
- Refund rate by new vs existing users
- Refund rate by transaction amount buckets
- Margin trend by product line
- Transaction success vs failure ratio
- Revenue before refunds vs after refunds

###  4. What executive summary would you present?
> In the past month, transaction volume increased 20%, but refund rate tripled from 2% to 6%, significantly eroding margins (22% → 16%). Initial analysis suggests that refund spikes are concentrated among new users and low-value merchant categories. This indicates potential issues in onboarding quality or merchant risk profiling.

If refund rates remain at 6%, projected quarterly profit impact could exceed X%. Immediate investigation into refund drivers and merchant/user segmentation is recommended

