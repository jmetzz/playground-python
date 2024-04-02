import pytest
from leetcode.p_1207_unique_number_of_occurrences import unique_occurrences_1, unique_occurrences_2


@pytest.mark.parametrize("func", [unique_occurrences_1, unique_occurrences_2])
@pytest.mark.parametrize(
    "input_arr, expected", [([1, 2], False), ([-3, 0, 1, -3, 1, 1, 1, -3, 10, 0], True), ([1, 2, 2, 1, 1, 3], True)]
)
def test_unique_occurrences(func, input_arr, expected):
    assert func(input_arr) == expected
