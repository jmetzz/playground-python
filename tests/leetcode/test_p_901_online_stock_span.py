from leetcode.p_901_online_stock_span import StockSpanner


def test_stock_spanner():
    prices_stream = [(100, 1), (80, 1), (60, 1), (70, 2), (60, 1), (75, 4), (85, 6)]
    stock_spanner = StockSpanner()
    for price, expected_span in prices_stream:
        assert stock_spanner.next(price) == expected_span
