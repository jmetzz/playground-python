import pytest
from leetcode.p_1043_partition_array_for_max_sum import PartitionArrayForMaxSum


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 15, 7, 9, 2, 5, 10], 3, 84),
        ([1, 4, 1, 5, 7, 3, 6, 1, 9, 9, 3], 4, 83),
        ([1], 1, 1),
        ([1, 2, 3, 4, 6], 3, 22),
    ],
)
def test_solve_dfs(test_input_arr, test_input_k, expected):
    assert PartitionArrayForMaxSum().solve_dfs(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 15, 7, 9, 2, 5, 10], 3, 84),
        ([1, 4, 1, 5, 7, 3, 6, 1, 9, 9, 3], 4, 83),
        ([1], 1, 1),
        ([1, 2, 3, 4, 6], 3, 22),
    ],
)
def test_solve_dfs_list_memo(test_input_arr, test_input_k, expected):
    assert PartitionArrayForMaxSum().solve_dfs_list_memo(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 15, 7, 9, 2, 5, 10], 3, 84),
        ([1, 4, 1, 5, 7, 3, 6, 1, 9, 9, 3], 4, 83),
        ([1], 1, 1),
        ([1, 2, 3, 4, 6], 3, 22),
    ],
)
def test_solve_dfs_dict_memo(test_input_arr, test_input_k, expected):
    assert PartitionArrayForMaxSum().solve_dfs_dict_memo(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 15, 7, 9, 2, 5, 10], 3, 84),
        ([1, 4, 1, 5, 7, 3, 6, 1, 9, 9, 3], 4, 83),
        ([1], 1, 1),
        ([1, 2, 3, 4, 6], 3, 22),
    ],
)
def test_solve_tabulation_top_down(test_input_arr, test_input_k, expected):
    assert PartitionArrayForMaxSum().solve_tabulation_top_down(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 15, 7, 9, 2, 5, 10], 3, 84),
        ([1, 4, 1, 5, 7, 3, 6, 1, 9, 9, 3], 4, 83),
        ([1], 1, 1),
        ([1, 2, 3, 4, 6], 3, 22),
    ],
)
def test_solve_tabulation_top_down_circular_list(test_input_arr, test_input_k, expected):
    assert PartitionArrayForMaxSum().solve_tabulation_top_down_circular_list(test_input_arr, test_input_k) == expected
