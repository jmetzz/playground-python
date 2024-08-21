import pytest

from leetcode.p_274_h_index import h_index_counting, h_index_sorting_ascending, h_index_sorting_descending


@pytest.mark.parametrize("func", [h_index_sorting_ascending, h_index_sorting_descending, h_index_counting])
@pytest.mark.parametrize("input_data, expected", [([3, 0, 6, 1, 5], 3), ([1, 3, 1], 1)])
def test_function_name(func, input_data, expected):
    assert func(input_data) == expected
