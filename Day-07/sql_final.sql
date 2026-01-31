-- Solve ALL (write, then explain in 1 line):
-- assume ads(campaign, clicks, impressions, cost)
-- CTR & CPC per campaign (zero-safe)
SELECT 
    campaign,
    CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END AS ctr,
    CASE WHEN clicks = 0 THEN 0 ELSE cost * 1.0 / clicks END AS cpc
FROM ads;

-- Campaigns with CTR above overall avg (SUBQUERY)
SELECT 
    campaign,
    CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END AS ctr
FROM ads
WHERE (CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END) > 
      (SELECT AVG(CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END) FROM ads);
-- Same using CTE
SELECT 
    campaign,
    ctr
FROM (
    SELECT 
        campaign,
        CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END AS ctr
    FROM ads
) AS sub
WHERE ctr > (SELECT AVG(ctr) FROM (
    SELECT 
        CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END AS ctr
    FROM ads
) AS avg_sub);
-- Best campaign (CTR high & CPC low)
SELECT 
    campaign,
    CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END AS ctr,
    CASE WHEN clicks = 0 THEN 0 ELSE cost * 1.0 / clicks END AS cpc
FROM ads
ORDER BY ctr DESC, cpc ASC
LIMIT 1;
-- Campaigns wasting spend (logic-based)
SELECT 
    campaign,
    cost
FROM ads
WHERE (CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END) < 0.01
   OR (CASE WHEN clicks = 0 THEN 0 ELSE cost * 1.0 / clicks END) > 10.0;
-- Where should budget be reallocated? (single query)
SELECT 
    campaign,
    cost,
    CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END AS ctr,
    CASE WHEN clicks = 0 THEN 0 ELSE cost * 1.0 / clicks END AS cpc
FROM ads
ORDER BY
    (CASE WHEN impressions = 0 THEN 0 ELSE clicks * 1.0 / impressions END) DESC,
    (CASE WHEN clicks = 0 THEN 0 ELSE cost * 1.0 / clicks END) ASC;
-- Bonus: Rank campaigns by efficiency (no window fn)
SELECT 
    a.campaign,
    a.cost,
    CASE WHEN a.impressions = 0 THEN 0 ELSE a.clicks * 1.0 / a.impressions END AS ctr,
    CASE WHEN a.clicks = 0 THEN 0 ELSE a.cost * 1.0 / a.clicks END AS cpc,
    (SELECT COUNT(*) + 1
     FROM ads b
     WHERE (CASE WHEN b.impressions = 0 THEN 0 ELSE b.clicks * 1.0 / b.impressions END) > 
           (CASE WHEN a.impressions = 0 THEN 0 ELSE a.clicks * 1.0 / a.impressions END)
        OR ((CASE WHEN b.impressions = 0 THEN 0 ELSE b.clicks * 1.0 / b.impressions END) = 
            (CASE WHEN a.impressions = 0 THEN 0 ELSE a.clicks * 1.0 / a.impressions END)
            AND (CASE WHEN b.clicks = 0 THEN 0 ELSE b.cost * 1.0 / b.clicks END) < 
                (CASE WHEN a.clicks = 0 THEN 0 ELSE a.cost * 1.0 / a.clicks END))
    ) AS efficiency_rank
FROM ads a
ORDER BY efficiency_rank ASC;