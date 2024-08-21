import pytest

from leetcode.p_14_longest_common_prefix import (
    longest_common_prefix__brute_force,
    longest_common_prefix__first_and_last,
    longest_common_prefix__horizontal_scanning,
    longest_common_prefix__vertical_scannning,
    longest_common_prefix__zip_vertical,
    longest_common_prefix_trie,
)

test_cases = [(["flower", "flow", "flight"], "fl"), (["dog", "racecar", "car"], "")]


@pytest.mark.parametrize(
    "func",
    [
        longest_common_prefix__brute_force,
        longest_common_prefix__vertical_scannning,
        longest_common_prefix__horizontal_scanning,
        longest_common_prefix__zip_vertical,
        longest_common_prefix__first_and_last,
        longest_common_prefix_trie,
    ],
)
@pytest.mark.parametrize("input_arr, expected", test_cases)
def test_longest_common_prefix(func, input_arr, expected):
    assert func(input_arr) == expected
