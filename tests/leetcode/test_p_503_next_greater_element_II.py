import pytest
from leetcode.p_503_next_greater_element_II import next_greater_elements_1, next_greater_elements_2


@pytest.mark.parametrize("func", [next_greater_elements_1, next_greater_elements_2])
@pytest.mark.parametrize(
    "input_arr, expected",
    [([1, 2, 1], [2, -1, 2]), ([1, 2, 3, 4, 3], [2, 3, 4, -1, 4]), ([2, 1, 2, 4, 3], [4, 2, 4, -1, 4])],
)
def test_next_greater_element(func, input_arr, expected):
    assert func(input_arr) == expected
