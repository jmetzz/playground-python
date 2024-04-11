-- individual sales profit in a given time period for a given list of products
SELECT 
    sp.id "provider_id", 
    sp.name "service_provider", 
    s.sale_date, 
    s.id "sales_id", 
    p.id "product_id", 
    p.name "product", 
    p.cost, 
    s.unit_price sales_price, 
    s.quantity, 
    sp.geo_id,
    s.unit_price * s.quantity revenue,
    s.unit_price * s.quantity - s.quantity * p.cost profit
FROM sales s
    right join product p on p.id = s.product_id
    join service_provider sp on s.service_provider_id = sp.id
where 
    p.id in (1,2,3,4,5,6)
    and s.sale_date BETWEEN date('now', '-15 days') AND date('now')
order by s.sale_date;


-- simplified sales profit by product in a given time period
SELECT 
    p.id "product_id", 
    p.name "product", 
    p.cost, 
    s.unit_price sales_price, 
    sum(s.quantity), 
    sum(s.unit_price * s.quantity) revenue,
    sum(s.unit_price * s.quantity - s.quantity * p.cost) profit
FROM 
    sales s
    right join product p on p.id = s.product_id
    join service_provider sp on s.service_provider_id = sp.id
WHERE 
    p.id in (1,2,3,4,5,6)
    and s.sale_date BETWEEN date('now', '-15 days') AND date('now')
group by p.id
order by s.sale_date;



-- simplified sales profit by product in a given time period,
-- and including service provider information
SELECT 
    s.id "sales_id",
    sp.id "provider_id", 
    sp.name "provider_name", 
    p.cost "unit_cost", 
    s.unit_price, 
    s.quantity, 
    s.unit_price * s.quantity sales_value, 
    (s.unit_price - p.cost) * s.quantity profit
FROM sales s 
    right join product p on p.id = s.product_id
    left join service_provider sp on sp.id = s.service_provider_id
WHERE
    p.id in (1,2,3,4,5,6)
    and s.sale_date BETWEEN date('now', '-15 days') AND date('now')
GROUP BY sp.id, p.id;



-- 
WITH profit_calc as (
    -- Computes the profit for each product for each service provider
    SELECT
        sp.id as service_provider_id,
        sp.name as service_provider_name,
        p.id as product_id,
        p.name as product_name,
        sum((s.unit_price - p.cost) * s.quantity) as sales_profit
    FROM sales s
    JOIN product p on s.product_id = p.id
    JOIN service_provider sp on s.service_provider_id = sp.id
    WHERE
        -- s.sale_date BETWEEN ? and ? -- placeholder for start_date and end_date
        -- and s.product_id in ({}) -- placeholder for the list of target product ids
        s.sale_date BETWEEN date('now', '-15 days') AND date('now')
        and s.product_id in (1,2,3,4,5,6)
    GROUP BY sp.id, p.id
),
most_profitable as (
    -- Assigns a rank to each product within each service provider based on profitability.
    SELECT
        service_provider_id,
        service_provider_name,
        product_id as most_profitable_product_id,
        product_name as most_profitable_product_name,
        sales_profit as most_profitable_profit
    FROM (
        SELECT *, 
            ROW_NUMBER() OVER (PARTITION BY service_provider_id ORDER BY sales_profit DESC) as rn
        FROM profit_calc
    ) ranked
    WHERE rn = 1
),
least_profitable as (
    -- Assigns a rank to each product within each service provider based on profitability.
    SELECT
        service_provider_id,
        service_provider_name,
        product_id as least_profitable_product_id,
        product_name as least_profitable_product_name,
        sales_profit as least_profitable_profit
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY service_provider_id ORDER BY sales_profit ASC) as rn
        FROM profit_calc
    ) ranked
    WHERE rn = 1
)

SELECT
    mp.service_provider_id,
    mp.service_provider_name,
    mp.most_profitable_product_id,
    mp.most_profitable_product_name,
    mp.most_profitable_profit,
    lp.least_profitable_product_id,
    lp.least_profitable_product_name,
    lp.least_profitable_profit
FROM most_profitable mp
JOIN least_profitable lp ON mp.service_provider_id = lp.service_provider_id
ORDER BY mp.service_provider_id;


-- second approach using only one ranking function
WITH profit_calc as (
    SELECT
        sp.id as service_provider_id,
        sp.name as service_provider_name,
        p.id as product_id,
        p.name as product_name,
        sum((s.unit_price - p.cost) * s.quantity) as sales_profit
    FROM sales s
    JOIN product p on s.product_id = p.id
    JOIN service_provider sp on s.service_provider_id = sp.id
    WHERE
        s.sale_date BETWEEN date('now', '-15 days') AND date('now')
        and s.product_id in (1,2,3,4,5,6)
    GROUP BY sp.id, p.id
),
ranked_profits as (
    -- calculates both the most and least profitable rankings in one go
    SELECT
        service_provider_id,
        service_provider_name,
        product_id,
        product_name,
        sales_profit,
        ROW_NUMBER() OVER (PARTITION BY service_provider_id ORDER BY sales_profit DESC) as rank_highest,
        ROW_NUMBER() OVER (PARTITION BY service_provider_id ORDER BY sales_profit ASC) as rank_lowest
    FROM profit_calc
)

SELECT
    rp.service_provider_id,
    rp.service_provider_name,
    rp.product_id AS most_profitable_product_id,
    rp.product_name AS most_profitable_product_name,
    rp.sales_profit AS most_profitable_profit,
    lp.product_id AS least_profitable_product_id,
    lp.product_name AS least_profitable_product_name,
    lp.sales_profit AS least_profitable_profit
FROM ranked_profits rp
JOIN ranked_profits lp ON rp.service_provider_id = lp.service_provider_id AND rp.rank_highest = 1 AND lp.rank_lowest = 1
ORDER BY rp.service_provider_id;




-- validate service provider 3 does have sales with the least and most profitable products
SELECT 
    s.id AS 'sales_id',
    p.name AS 'product_name',  -- This is the name of the attraction
    p.id, 
    s.quantity,
    s.unit_price,
    s.sale_date,
    p.cost,
    s.unit_price * s.quantity revenue,
    s.unit_price * s.quantity - s.quantity * p.cost profit
FROM 
    sales s
JOIN 
    product p ON s.product_id = p.id
WHERE 
    s.service_provider_id = 3  -- Filtering sales for service provider with ID 3
    and p.id in (1, 3)
ORDER BY p.id;




-- Tie resolver by returning the list of product id with the same profit
-- It possible to return a list of product IDs for both the most and least profitable products
-- in cases where there are ties in profit, while still keeping one line per service provider. 
-- This can be achieved using the STRING_AGG function (or equivalent depending on your DBMS
-- like GROUP_CONCAT in MySQL, STRING_AGG in PostgreSQL, or a similar function in other databases)
-- to aggregate product IDs into a list. 
-- However, SQLite does not natively support a built-in function for string aggregation 
-- until recent versions (SQLite 3.30.0 and above introduced GROUP_CONCAT as an enhancement).
-- The query assuming a function like STRING_AGG or GROUP_CONCAT is available your SQL environment:

WITH profit_calc AS (
    SELECT
        sp.id AS service_provider_id,
        sp.name AS service_provider_name,
        p.id AS product_id,
        p.name AS product_name,
        (s.unit_price - p.cost) * s.quantity AS individual_profit
    FROM sales s
    JOIN product p ON s.product_id = p.id
    JOIN service_provider sp ON s.service_provider_id = sp.id
    WHERE
        s.sale_date BETWEEN date('now', '-15 days') AND date('now')
        AND s.product_id IN (1,2,3,4,5,6)
),
aggregated_profits AS (
    SELECT
        service_provider_id,
        service_provider_name,
        product_id,
        product_name,
        SUM(individual_profit) AS total_profit
    FROM profit_calc
    GROUP BY service_provider_id, product_id
),
ranked_profits AS (
    SELECT
        service_provider_id,
        service_provider_name,
        product_id,
        product_name,
        total_profit,
        RANK() OVER (PARTITION BY service_provider_id ORDER BY total_profit DESC) AS rank_highest,
        RANK() OVER (PARTITION BY service_provider_id ORDER BY total_profit ASC) AS rank_lowest
    FROM aggregated_profits
)
SELECT
    service_provider_id,
    service_provider_name,
    STRING_AGG(CASE WHEN rank_highest = 1 THEN product_id END, ', ') AS most_profitable_products,
    STRING_AGG(CASE WHEN rank_lowest = 1 THEN product_id END, ', ') AS least_profitable_products
FROM ranked_profits
GROUP BY service_provider_id, service_provider_name
ORDER BY service_provider_id;
