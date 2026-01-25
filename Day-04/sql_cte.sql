CREATE TABLE ads (
    ad_id INT PRIMARY KEY,
    campaign VARCHAR(20),
    clicks INT,
    impressions INT,
    cost INT
);
INSERT INTO ads (ad_id, campaign, clicks, impressions, cost)
VALUES (1, 'Summer_Sale', 120, 5000, 250),
    (2, 'Summer_Sale', 95, 4200, 210),
    (3, 'Summer_Sale', 60, 3000, 150),
    (4, 'App_Launch', 200, 8000, 400),
    (5, 'App_Launch', 180, 7600, 380),
    (6, 'App_Launch', 90, 5000, 220),
    (7, 'Festive_Offer', 300, 9000, 500),
    (8, 'Festive_Offer', 260, 8500, 460),
    (9, 'Festive_Offer', 150, 6000, 300),
    (10, 'Brand_Awareness', 40, 7000, 180),
    (11, 'Brand_Awareness', 55, 7500, 200);
SELECT *
from ads;
-- CTR per campaign
WITH CTR as (
    SELECT campaign,
        (sum(clicks)) as clicks,
        sum(impressions) as impressions
    FROM ads
    GROUP BY campaign
)
SELECT campaign,
    (clicks * 1.0 / impressions) * 100 as CTR
from CTR;
-- Campaign with highest CTR
WITH CTR as (
    SELECT campaign,
        (sum(clicks)) as clicks,
        sum(impressions) as impressions
    FROM ads
    GROUP BY campaign
)
SELECT campaign,
    max((clicks * 1.0 / impressions)) as CTR
from CTR;
-- Avg cost per click per campaign
WITH CTR as (
    SELECT campaign,
        (sum(clicks)) as clicks,
        sum(cost) as cost
    FROM ads
    GROUP BY campaign
)
SELECT campaign,
    cost / clicks as costperclick
from ctr;
-- Campaigns with CTR above overall avg (SUBQUERY)
-- WITH CTR as (
--     SELECT campaign, (sum(clicks)) as clicks,sum(impressions) as impressions
--     FROM ads
--     GROUP BY campaign
-- ),
--  avg_ctr as(
--     SELECT campaign, (clicks*1.0/impressions) as CTR1
--     from CTR
-- )
-- select * from avg_ctr
-- WHERE CTR1 >(
-- SELECT avg(CTR1) from avg_ctr);
-- A simpler and more optimal version:
WITH campaign_ctr as (
    SELECT campaign,
        SUM(clicks) * 1.0 / SUM(impressions) as ctr
    from ads
    GROUP BY campaign
)
SELECT *
FROM campaign_ctr
WHERE ctr > (
        SELECT AVG(ctr)
        FROM campaign_ctr
    );
-- With sub-query
SELECT campaign,
    ctr
FROM (
        SELECT campaign,
            SUM(clicks) * 1.0 / SUM(impressions) AS ctr
        FROM ads
        GROUP BY campaign
    ) t
WHERE ctr > (
        SELECT AVG(ctr)
        FROM (
                SELECT SUM(clicks) * 1.0 / SUM(impressions) AS ctr
                FROM ads
                GROUP BY campaign
            ) x
    );
-- Rank campaigns by clicks (no window fn yet)
SELECT campaign,
    clicks
from ads
GROUP BY campaign
order by clicks desc;
-- Total cost share (%) per campaign
SELECT campaign,
    SUM(cost) * 100.0 / (
        SELECT SUM(cost)
        FROM ads
    ) AS cost_share_pct
from ads
GROUP BY campaign;