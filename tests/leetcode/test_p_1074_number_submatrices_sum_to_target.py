import pytest
from leetcode.p_1074_number_submatrices_sum_to_target import NumSubmatricesSumTarget


@pytest.mark.parametrize(
    "test_input_matrix, test_input_target, expected",
    [
        ([[904]], 0, 0),
        ([[1, -1], [-1, 1]], 0, 5),
        ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 0, 4),
    ],
)
def test_number_submatrices_sum_target_grid(test_input_matrix, test_input_target, expected):
    assert NumSubmatricesSumTarget().solve_2d_grid(test_input_matrix, test_input_target) == expected


@pytest.mark.parametrize(
    "test_input_matrix, test_input_target, expected",
    [
        ([[904]], 0, 0),
        ([[1, -1], [-1, 1]], 0, 5),
        ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 0, 4),
    ],
)
def test_number_submatrices_sum_target_hashmap(test_input_matrix, test_input_target, expected):
    assert NumSubmatricesSumTarget().solve_hashmap(test_input_matrix, test_input_target) == expected
