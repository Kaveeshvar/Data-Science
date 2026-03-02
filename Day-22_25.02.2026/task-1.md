# How I Would Analyze a Drop in Payment Success Rate in a Digital Wallet

### What is payment success rate -> The payment success rate (PSR) is the percentage of total customer payment attempts that are successfully authorized and completed without errors.
Why this metric is VERY important (especially in fintech interviews)

Because payment success rate directly impacts:

    Revenue

    Customer experience

    Trust

    Repeat usage

    Merchant satisfaction

What counts as “failure”?

Failures can happen due to:

    Bank downtime

    UPI timeout

    Network issues

    User entered wrong OTP

    Insufficient balance

    Fraud detection blocks

    Gateway failure

    3DS authentication failure

How to Improve Payment Success Rate:
Implement Retry Logic: Automatically retry failed transactions in real-time.
Use Multiple Gateways: Utilize routing to switch to a better-performing gateway if one is down.
Simplify Checkout: Reduce steps, offer saved cards, and provide diverse payment options to reduce user friction.

Payment Success Rate is a funnel metric.

**Attempt → Authentication → Authorization → Capture → Success**

You can calculate drop-offs at each stage.

How would you improve PSR?
    Analyze failure codes distribution

    Identify high-failure banks/gateways

    Enable smart routing (switch gateway dynamically)

    Retry logic for technical failures

    Improve UX for OTP entry

    Real-time monitoring dashboard

    Work with partner banks for SLA improvement

How it connects to revenue + trust -> 