import pytest

from leetcode.p_15_three_sum import three_sum_sorting, three_sum_speedup


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([0, 0, 0], {(0, 0, 0)}),
        ([0, 1, 1], set()),
        ([-1, 0, 1, 2, -1, -4], {(-1, -1, 2), (-1, 0, 1)}),
        ([-2, 0, 1, 1, 2], {(-2, 0, 2), (-2, 1, 1)}),
    ],
)
def test_three_sum_sorting(input_data, expected):
    assert three_sum_sorting(input_data) == expected


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([0, 0, 0], [[0, 0, 0]]),
        ([0, 0, 0, 0], [[0, 0, 0]]),
        ([0, 1, 1], []),
        ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ([-2, 0, 1, 1, 2], [[-2, 0, 2], [-2, 1, 1]]),
        (
            [-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4],
            [
                [-4, 0, 4],
                [-4, 1, 3],
                [-3, -1, 4],
                [-3, 0, 3],
                [-3, 1, 2],
                [-2, -1, 3],
                [-2, 0, 2],
                [-1, -1, 2],
                [-1, 0, 1],
            ],
        ),
    ],
)
def test_three_sum_speedup(input_data, expected):
    assert three_sum_speedup(input_data) == expected
