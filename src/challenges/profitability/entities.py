from dataclasses import dataclass


@dataclass
class Product:
    id: str
    name: str
    description: str
    category: str
    geo_id: float
    cost: float


@dataclass
class Geo:
    id: int
    country: str
    region: str


@dataclass
class Sales:
    sales_id: str
    product_id: str
    product: str
    unit_cost: int
    unit_sales_price: int
    quantity: int
