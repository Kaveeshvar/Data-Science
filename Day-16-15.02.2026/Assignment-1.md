# Assignment - 1

You are hired by a fintech startup that processes digital payments.

Last month:

- Revenue grew 8%

- Active users grew 15%

- But profits dropped 4%

- No dataset.

- Just think.

### Questions:

1. What metrics do you investigate first?
> Revenue = Price × Volume
> Profit = Revenue – Variable Costs – Fixed Costs
That means 
- Revenue per user likely decreased
- Costs increased disproportionately
- customer acquisition cost spiked


2. What hypotheses do you form?

- New users have lower ARPU (Average Revenue Per User) than existing users
- Discounts/cashbacks increased to drive growth
- Transaction fees decreased
- Payment failure retries increased infra cost
- Fraud losses increased
- Marketing CAC (Customer Acquisition Cost) spiked

3. What SQL tables would you expect?
> users, transactions, merchants, fees, refunds, chargebacks, marketing_spend, infra_costs, subscription_plans

4. What dashboard sections would you design?
- Executive Layer : 
    Revenue
    Profit
    Margin %
    ARPU
    CAC ( Customer Acquisition Cost)
    Active Users(new vs returning)

- Diagnostic Layer : 
    Revenue by user Cohort
    Revenue by merchant category
    Cost Breakdown (infra vs marketing vs fraud)
    Refund Rate
    Chargeback rate

- Root Cause Layer : 
    New user ARPU vs Existing ARPU
    Discount usage %
    Transaction failure rate
    Cost per transaction trend

5. What could be happening?
- Growth driven by heavy cashback campaigns
- Lower transaction fees negotiated for big merchants
- High-value users churned
- Fraud increased
- Infrastructure scaling costs rose with user spike
- Payment retry loops increased server load
- Marketing CAC grew faster than LTV ( Loan-to-Value)


#### Revenue ↑ 8% 
Due to price or volume?
Due to new users or existing?
Due to seasonality?
Due to product mix?

#### Profit ↓ 4%
Variable cost spike? ( A sudden increase in costs that change with production.) 
Fixed cost spike?
One-time event?
Fraud/Refunds?
