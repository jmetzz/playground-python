SALES_DETAILS = """
SELECT 
    s.id "sales_id", 
    p.id "product_id", 
    p.name "product", 
    sp.id "provider_id", 
    p.cost "unit_cost", 
    s.unit_price "unit_sales_price",
    s.quantity,
    sp.name "service_provider", 
    s.sale_date, 
    sp.geo_id
FROM sales s
    right join product p on p.id = s.product_id
    join service_provider sp on s.service_provider_id = sp.id
WHERE
    p.id in ({})
    and s.sale_date BETWEEN ? AND ?
ORDER BY s.sale_date;
"""
