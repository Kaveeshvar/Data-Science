SELECT *
from ads;
-- Campaigns with CTR above overall average
SELECT campaign,
    SUM(clicks) * 1.0 / SUM(impressions) AS ctr
from ads
group BY campaign
HAVING sum(clicks) * 1.0 / sum(impressions) >(
        SELECT (sum(clicks) * 1.0 / sum(impressions)) as avg_ctr
        from ads
    );
-- Top 2 campaigns by CTR
SELECT campaign,
    sum(clicks) * 1.0 / sum(impressions) as ctr
from ads
group by campaign
ORDER BY ctr desc
LIMIT 2;
-- Campaigns where CPC is worse than overall avg CPC
SELECT campaign,
    SUM(cost) * 1.0 / SUM(clicks) AS cpc
from ads
group BY campaign
HAVING sum(cost) * 1.0 / sum(clicks) <(
        SELECT (sum(cost) * 1.0 / sum(clicks)) as avg_ctr
        from ads
    );
-- Best campaign considering BOTH CTR high & CPC low
with kpi as (
    Select campaign,
        (sum(clicks) * 1.0 / sum(impressions)) AS ctr,
        (sum(cost) * 1.0 / sum(clicks)) AS cpc
    from ads
    group by campaign
)
Select *
from kpi
ORDER BY (ctr / cpc) desc
LIMIT 1;
-- Rank campaigns by efficiency (no window fn)
with kpi as (
    Select campaign,
        (sum(clicks) * 1.0 / NULLIF(sum(impressions), 0)) AS ctr,
        (sum(cost) * 1.0 / NULLIF(sum(clicks), 0)) AS cpc
    from ads
    group by campaign
)
Select *
from kpi
ORDER BY (ctr / cpc) desc;
-- Rewrite #1 - Campaigns with CTR above overall average using a CTE
with campaign_ctr as (
    Select campaign,
        sum(clicks) * 1.0 / nullif(sum(impressions), 0) as ctr
    from ads
    group by campaign
),
overall_avg as (
    select avg(ctr) as avg_ctr
    from campaign_ctr
)
select c.*
from campaign_ctr c
    cross join overall_avg o
where c.ctr > o.avg_ctr;
-- One query of your own (business-driven)
SELECT campaign ,sum(clicks) as clicks_sum,sum(impressions) as impressions_sum,
(((sum(clicks)*1.0 / sum(impressions) ) *100) / ((sum(cost)*1.0 / sum(clicks) ) *100 )) as ctrlcpc
from ads
group by campaign
ORDER BY ctrlcpc desc;