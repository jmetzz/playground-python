import pytest

from leetcode.p_629_k_inverse_pairs_array import KInversePairsArray


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # (n, k), expected
        (
            (3, 0),
            1,
        ),  # Only the array [1,2,3] which consists of 1 to 3 has exactly 0 inverse pairs
        ((3, 1), 2),  # The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.
    ],
)
def test_k_inverse_pairs_array_memo(test_input: tuple, expected: int):
    assert KInversePairsArray().solve_memo(test_input[0], test_input[1], {}) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # (n, k), expected
        # ((3, 0), 1),  # Only the array [1,2,3] which consists of 1 to 3 has exactly 0 inverse pairs
        ((3, 1), 2),  # The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.
    ],
)
def test_k_inverse_pairs_array_full_grid(test_input: tuple, expected: int):
    assert KInversePairsArray().solve_dp_full_grid(test_input[0], test_input[1]) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # (n, k), expected
        (
            (3, 0),
            1,
        ),  # Only the array [1,2,3] which consists of 1 to 3 has exactly 0 inverse pairs
        ((3, 1), 2),  # The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.
    ],
)
def test_k_inverse_pairs_array_shorter_grid(test_input: tuple, expected: int):
    assert KInversePairsArray().solve_dp_short_grid(test_input[0], test_input[1]) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # (n, k), expected
        (
            (3, 0),
            1,
        ),  # Only the array [1,2,3] which consists of 1 to 3 has exactly 0 inverse pairs
        ((3, 1), 2),  # The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.
    ],
)
def test_k_inverse_pairs_array_shorter_sliding_window(test_input: tuple, expected: int):
    assert KInversePairsArray().solve_dp_short_sliding_window(test_input[0], test_input[1]) == expected
