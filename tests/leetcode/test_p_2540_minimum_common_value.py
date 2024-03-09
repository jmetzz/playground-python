import pytest
from leetcode.p_2540_minimum_common_value import (
    get_common__with_bin_search,
    get_common__with_loop_and_set,
    get_common__with_set_intersection,
    get_common__with_two_pointers,
)

INPUT_PARAMS = [
    ([1, 2, 3], [2, 4], 2),
    ([2, 3, 4, 5, 6, 7], [3, 3, 3, 3, 4, 7], 3),
    ([2, 3, 4, 5, 6, 7, 10], [7, 8, 9, 10], 7),
    ([1], [7, 8, 9, 10], -1),
    ([], [], -1),
]


@pytest.mark.parametrize("input_arr_1, input_arr_2, expected", INPUT_PARAMS)
def test_get_common_approach_1(input_arr_1, input_arr_2, expected):
    assert get_common__with_set_intersection(input_arr_1, input_arr_2) == expected


@pytest.mark.parametrize("input_arr_1, input_arr_2, expected", INPUT_PARAMS)
def test_get_common_approach_2(input_arr_1, input_arr_2, expected):
    assert get_common__with_loop_and_set(input_arr_1, input_arr_2) == expected


@pytest.mark.parametrize("input_arr_1, input_arr_2, expected", INPUT_PARAMS)
def test_get_common_approach_3(input_arr_1, input_arr_2, expected):
    assert get_common__with_bin_search(input_arr_1, input_arr_2) == expected


@pytest.mark.parametrize("input_arr_1, input_arr_2, expected", INPUT_PARAMS)
def test_get_common_approach_4(input_arr_1, input_arr_2, expected):
    assert get_common__with_two_pointers(input_arr_1, input_arr_2) == expected
