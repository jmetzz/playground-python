import pytest

from leetcode.p_442_find_all_duplicates_in_an_array import find_duplicates, find_duplicates_space_optimized


@pytest.mark.parametrize("func", [find_duplicates, find_duplicates_space_optimized])
@pytest.mark.parametrize(
    "input_list, expected",
    [
        # ([], []),
        # ([1], []),
        # ([1, 2, 3, 4, 5], []),
        # ([1,1,2], [1]),
        # ([1, 2, 3, 2, 4], [2]),
        ([4, 3, 2, 7, 8, 2, 3, 1], [2, 3]),
        ([1, 1, 1, 1, 1], [1]),
    ],
)
def test_find_duplicates(func, input_list, expected):
    assert sorted(func(input_list)) == sorted(expected)
