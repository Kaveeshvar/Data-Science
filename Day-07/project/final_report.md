# Problem statement (3–4 lines)
Advertising campaigns generate a lot of data, and analyzing this data efficiently is crucial for optimizing performance. The goal is to filter campaigns based on their click-through rates (CTR) and create a mapping of campaigns to their CTRs using Python comprehensions.

# Data overview (rows, columns, caveats)
The dataset consists of 	ad_id	campaign	clicks	impressions	cost. The key columns for this analysis are 'campaign', 'clicks', and 'impressions'. The dataset may contain campaigns with zero impressions, which need to be handled to avoid division by zero errors.

# Metrics defined (CTR, CPC — why they matter)
- Click-Through Rate (CTR): CTR is calculated as (clicks / impressions). It measures the effectiveness of an ad campaign in generating clicks from impressions. A higher CTR indicates a more successful campaign.
- Cost Per Click (CPC): CPC is calculated as (cost / clicks). It indicates how much is spent on average for each click. Lower CPC values are generally preferred as they indicate cost efficiency.

# 3 questions answered (with results):

## Which campaign to scale?
We should scale campaigns with a CTR above a certain threshold (e.g., 0.05 or 5%). These campaigns are performing well and have the potential to generate more clicks and conversions if given more budget.
It is Festive_Offer.

## Which to pause?
We should consider pausing campaigns with a CTR below a certain threshold (e.g., 0.01 or 1%). These campaigns are underperforming and may not be worth the investment.
It is Brand_Awareness.

## What to change next?
Campaigns with moderate CTR (between 0.01 and 0.05) should be analyzed further to identify areas for improvement. This could involve changing ad creatives, targeting different audiences, or adjusting bids to enhance performance.
They are Summer_Sale and App_Launch.

## Recommendations (bullet points, action-oriented)
- Scale the "Festive_Offer" campaign by increasing its budget and expanding its reach to maximize its high CTR.
- Pause the "Brand_Awareness" campaign to reallocate resources to more effective campaigns.
- Analyze the "Summer_Sale" and "App_Launch" campaigns to identify potential improvements in ad creatives, targeting strategies, or bidding approaches to boost their CTR.
