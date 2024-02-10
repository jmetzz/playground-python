import pytest

from dynamic_programming import p_3_can_sum as can_sum

@pytest.mark.parametrize("test_input_target, test_input_arr, expected",
                         [
                             (7, [7, 2], True),
                             (7, [5, 3, 4, 7], True),
                             (7, [2, 4], False),
                             (8, [2, 3, 5], True),
                             (100, [7, 14], False),
                         ])

def test_can_sum_recursive(test_input_target, test_input_arr, expected):
    assert can_sum.solve_recursive(test_input_target, test_input_arr) == expected


@pytest.mark.parametrize("test_input_target, test_input_arr, expected",
                         [
                             (7, [7, 2], True),
                             (7, [5, 3, 4, 7], True),
                             (7, [2, 4], False),
                             (8, [2, 3, 5], True),
                             (3000, [7, 14], False),
                         ])
def test_can_sum_dp(test_input_target, test_input_arr, expected):
    assert can_sum.solve_memoization(test_input_target, test_input_arr, {}) == expected
