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

SELECT campaign, sum(clicks)*1.0/sum(impressions) from ads
 GROUP BY campaign;

-- Campaigns with CTR > overall average (SUBQUERY)
SELECT campaign,sum(clicks)*1.0/sum(impressions) as CTR
 from ads
 GROUP BY campaign
 HAVING CTR>(
    SELECT sum(clicks)*1.0/sum(impressions)
    from ads
 ) ;

-- Same as #1 using CTE
WITH campaign_ctr as
(
    SELECT campaign,
    sum(clicks)*1.0/sum(impressions) as ctr
     from ads
     group by campaign
),
overall_ctr as
(
    Select avg(ctr) as avg_ctr 
    from campaign_ctr
)
SELECT * from campaign_ctr c
cross join overall_ctr o
where c.ctr > o.avg_ctr;

-- Campaign with lowest CPC
SELECT campaign, sum(cost)*1.0/sum(clicks) as cpc
 from ads
 GROUP BY campaign
 ORDER BY cpc
 LIMIT 1;


--  Rank campaigns by efficiency (CTR desc, CPC asc)
with kpis as (
    select campaign,
    sum(clicks)*1.0/sum(impressions) as ctr,
    sum(cost)*1.0/sum(clicks) as cpc,
    (((sum(clicks)*1.0 / sum(impressions) ) *100) / 
    ((sum(cost)*1.0 / sum(clicks) ) *100 )) as ctrlcpc
    from ads
    group by campaign
)
select campaign,ctrlcpc*100 as efficiency from kpis
order by (ctr/cpc) desc; 

-- Identify campaigns wasting spend (logic-based)
SELECT campaign,
       SUM(clicks) * 1.0 / SUM(impressions) AS ctr,
       SUM(cost) * 1.0 / SUM(clicks) AS cpc
FROM ads
GROUP BY campaign
HAVING
    SUM(clicks) * 1.0 / SUM(impressions) < (
        SELECT SUM(clicks) * 1.0 / SUM(impressions) FROM ads
    )
AND
    SUM(cost) * 1.0 / SUM(clicks) > (
        SELECT SUM(cost) * 1.0 / SUM(clicks) FROM ads
    );

-- One query answering: “Where should we reallocate budget?”
SELECT campaign,
       SUM(clicks) * 1.0 / SUM(impressions) AS ctr,
       SUM(cost) * 1.0 / SUM(clicks) AS cpc
FROM ads
GROUP BY campaign
ORDER BY ctr DESC, cpc ASC;

-- Sorting by CTR ↓ and CPC ↑ (inverse) ranks campaigns where budget should be reallocated and increased first.
