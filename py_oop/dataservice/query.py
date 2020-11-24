sales_table_query = """
SELECT group_code, country_code, period_seq, item_id, projected_revenue_local FROM sales.item_sales
WHERE periodicity = 'Weekly'
AND period_seq = %(period_seq)s
AND projected_revenue_local > 0
"""
get_all_market_config = """
SELECT country_code,
    item_group_code,
    market_configuration->'app'->'low_price_cut_off' AS low_price_cut_off,
    market_configuration->'app'->'high_price_cut_off' AS high_price_cut_off,
    market_configuration->'app'->'medium_price_cut_off' AS medium_price_cut_off,
    market_configuration->'app'->'lower_price_cut_off' AS lower_price_range_cut_off,
    market_configuration->'app'->'upper_price_cut_off' AS upper_price_range_cut_off
FROM sales.item_group_countries
"""

get_market_config = """
SELECT country_code,
    item_group_code,
    market_configuration->'app'->'low_price_cut_off' AS low_price_cut_off,
    market_configuration->'app'->'high_price_cut_off' AS high_price_cut_off,
    market_configuration->'app'->'medium_price_cut_off' AS medium_price_cut_off,
    market_configuration->'app'->'lower_price_cut_off' AS lower_price_range_cut_off,
    market_configuration->'app'->'upper_price_cut_off' AS upper_price_range_cut_off
FROM sales.item_group_countries
WHERE country_code = %(country_code)s
AND item_group_code = %(item_group_code)s
"""

get_last_period_seq_per_cell = """
select item_group_code, country_code, max(period_seq) as period_seq
from sales.market_sales
group by item_group_code, country_code;
"""


get_base_facts: str = """
SELECT
    s.item_id,
    i.brand_name AS brand,
    s.country_code,
    s.item_group_code,
    s.periodicity,
    s.period_seq,
    projected_units > 0 AS sufficient_sales,
    s.projected_revenue_local AS revenue_local,
    s.projected_units AS salesunits,
    CASE s.projected_units
       WHEN 0 THEN NULL::numeric
       ELSE s.projected_revenue_local / s.projected_units
    END AS price,
    d.weighted_distribution AS wgt_distr,
    cast(1 AS float) AS LOC_price,
    cast(1 AS float) AS LOC_salesunits,
    cast(1 AS float) AS LOC_wgt_distr,
    count(period_seq) OVER (PARTITION BY item_id) AS no_of_periods_in_focus
FROM
    sales.item_sales_by_outletgroup s
LEFT JOIN
    sales.item_distribution_by_outletgroup d USING (
        item_id, country_code, item_group_code, periodicity, period_seq, outlet_group_code
    )
JOIN sales.item i USING (item_id, country_code)
WHERE
    country_code = %(country_code)s AND
    item_group_code = %(item_group_code)s AND
    period_seq BETWEEN %(period_seq)s - 4 AND %(period_seq)s AND
    periodicity = 'Weekly' AND
    outlet_group_code = 'ALL'
"""

get_moc: str = """
SELECT
    item_id,
    item_group_code,
    country_code,
    cast(1 AS float) AS loc_distance_euclidean,
    competitor_item_id AS competitor_item_id,
    e_distance AS distance_euclidean,
    distribution_overlap AS distribution_overlap,
    rank() OVER (PARTITION BY item_id, period_seq ORDER BY e_distance) AS my_rank
FROM
    algo.competitors
WHERE
    country_code = %(country_code)s AND
    item_group_code = %(item_group_code)s AND
    period_seq = %(period_seq)s AND
    periodicity = 'Weekly' AND
    (distribution_overlap >= 0.5 OR distribution_overlap IS NULL)
"""

get_tpr = """
SELECT
    item_id,
    efficiency AS tpr_efficiency_own,
    cast(1 AS float) AS loc_tpr_efficiency_own
FROM
    algo.temporary_price_reduction_aggregate
WHERE
    country_code = %(country_code)s AND
    item_group_code = %(item_group_code)s AND
    period_seq = %(period_seq)s AND
    periodicity = 'Weekly'
"""


get_all_data: str = """
WITH moc as (
  select item_id,
         item_group_code,
         country_code,
         cast(1 as float)                                                   as loc_distance_euclidean,
         competitor_item_id                                                 as competitor_item_id,
         e_distance                                                         as distance_euclidean,
         distribution_overlap                                               as distribution_overlap,
         rank() over (partition by item_id, period_seq order by e_distance) as my_rank
  from algo.competitors
  where country_code = %(country_code)s
    and item_group_code = %(item_group_code)s
    and period_seq in (%(period_seq)s)
    and periodicity = 'Weekly'
    and (distribution_overlap >= 0.5 or distribution_overlap IS NULL)
),

bf as (
  select item_id,
         item_group_code,
         country_code,
         period_seq,
         brand_name                                    as brand,
         average_price                                 as price,
         cast(1 as float)                              as LOC_price,
         sales_units                                   as salesunits,
         cast(1 as float)                              as LOC_salesunits,
         sufficient_sales,
         weighted_distribution                         as wgt_distr,
         cast(1 as float)                              as LOC_wgt_distr,
         count(period_seq) over (partition by item_id) as no_of_periods_in_focus
  from algo.base_facts
  where country_code = %(country_code)s
    and item_group_code = %(item_group_code)s
    and period_seq between %(period_seq)s - 4 and %(period_seq)s
    and periodicity = 'Weekly'
),

tpr as (
  select item_id,
         efficiency,
         cast(1 as float) as loc_tpr_efficiency
  from algo.temporary_price_reduction_aggregate
  where country_code = %(country_code)s
    and item_group_code = %(item_group_code)s
    and period_seq = %(period_seq)s
    and periodicity = 'Weekly'
)

select bf_self.item_id,
       bf_self.item_group_code,
       bf_self.country_code,
       bf_self.period_seq,
       moc.my_rank,
       moc.competitor_item_id,
       moc.loc_distance_euclidean,
       moc.distance_euclidean,
       moc.distribution_overlap,
       bf_self.brand                    as brand,
       bf_self.price                    as price_own,
       bf_self.loc_price                as LOC_price_own,
       bf_self.salesunits               as salesunits_own,
       bf_self.loc_salesunits           as LOC_salesunits_own,
       bf_self.sufficient_sales         as sufficient_sales_own,
       bf_self.wgt_distr                as wgt_distr_own,
       bf_self.LOC_wgt_distr            as LOC_wgt_distr_own,
       bf_comp.brand                    as brand_competitor,
       bf_comp.price                    as price_competitor,
       bf_comp.LOC_price                as LOC_price_competitor,
       bf_comp.salesunits               as salesunits_competitor,
       bf_comp.loc_salesunits           as LOC_salesunits_competitor,
       bf_comp.wgt_distr                as wgt_distr_competitor,
       bf_comp.LOC_wgt_distr            as LOC_wgt_distr_competitor,
       bf_self.no_of_periods_in_focus,
       tpr.efficiency                   as tpr_efficiency_own,
       tpr.loc_tpr_efficiency           as loc_tpr_efficiency_own
from moc
  right join bf as bf_self
    on moc.item_id = bf_self.item_id
  left join bf as bf_comp
    on moc.competitor_item_id = bf_comp.item_id
    and bf_self.period_seq = bf_comp.period_seq
  left join tpr
    on moc.item_id = tpr.item_id
WHERE
    bf_self.price <> 0
order by item_id, period_seq, competitor_item_id;
"""
