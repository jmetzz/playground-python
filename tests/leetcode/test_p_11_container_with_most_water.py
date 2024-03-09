import pytest
from leetcode.p_11_container_with_most_water import max_area


@pytest.mark.parametrize("input_arr, expected", [([1, 8, 6, 2, 5, 4, 8, 3, 7], 49), ([1, 1], 1)])
def test_max_volume(input_arr, expected):
    assert max_area(input_arr) == expected
