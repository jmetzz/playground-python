import pytest

from leetcode.p_787_cheapest_flights_within_k_stops import CheapestFlights


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (([[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]], 4, 0, 3, 1), 700),
        (([[0, 1, 100], [1, 2, 100], [0, 2, 500]], 3, 0, 2, 1), 200),
        (([[0, 1, 100], [1, 2, 100], [0, 2, 500]], 3, 0, 2, 0), 500),
    ],
)
def test_cheapest_flights(test_input, expected):
    flights, n, src, dst, k = test_input
    assert CheapestFlights().solve(n, flights, src, dst, k) == expected
