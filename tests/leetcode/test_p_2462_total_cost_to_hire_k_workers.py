import pytest
from leetcode.p_2462_total_cost_to_hire_k_workers import total_cost_one_heap, total_cost_two_heaps


@pytest.mark.parametrize("func", [total_cost_one_heap, total_cost_two_heaps])
@pytest.mark.parametrize(
    "cost_arr, k, candidates, expected", [([17, 12, 10, 2, 7, 2, 11, 20, 8], 3, 4, 11), ([1, 2, 4, 1], 3, 3, 4)]
)
def test_total_cost_of_hiring(func, cost_arr, k, candidates, expected):
    assert func(cost_arr, k, candidates) == expected
