import pytest
from leetcode.p_2300_successful_pairs_of_spells_and_potions import successful_pairs


@pytest.mark.parametrize(
    "spells, potions, success, expected",
    [([5, 1, 3], [1, 2, 3, 4, 5], 7, [4, 0, 3]), ([3, 1, 2], [8, 5, 8], 16, [2, 0, 2])],
)
def test_successful_pairs(spells, potions, success, expected):
    assert successful_pairs(spells, potions, success) == expected
