import pytest

from leetcode.p_026_remove_duplicates import DuplicatesRemover


@pytest.mark.parametrize("test_input, expected_arr",
                         [
                             ([1], [1]),
                             ([1, 1], [1]),
                             ([1, 2], [1, 2]),
                             ([1, 1, 2], [1, 2]),
                             ([0, 0, 1, 1, 1, 2, 2, 3, 3, 4], [0, 1, 2, 3, 4]),
                         ])
def test_remove_duplicates(test_input, expected_arr):
    actual_k = DuplicatesRemover().solve(test_input)

    assert actual_k == len(expected_arr)
    assert test_input[:actual_k] == expected_arr
