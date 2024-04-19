import pytest
from leetcode.p_1431_kids_with_the_greatest_number_of_candies import kids_with_candies


@pytest.mark.parametrize(
    "candies_array, extra_candies, expected",
    [
        ([2, 3, 5, 1, 3], 3, [True, True, True, False, True]),
        ([4, 2, 1, 1, 2], 1, [True, False, False, False, False]),
        ([12, 1, 12], 10, [True, False, True]),
    ],
)
def test_kids_with_candies(candies_array, extra_candies, expected):
    assert kids_with_candies(candies_array, extra_candies) == expected
