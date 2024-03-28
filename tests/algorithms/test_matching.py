import pytest
from algorithms.strings.matching import compute_lps_array, compute_lps_array_2, compute_lps_array_3

test_cases = [
    ("", []),
    ("AAAAA", [0, 1, 2, 3, 4]),
    ("ABCDE", [0, 0, 0, 0, 0]),
    ("AAABAAA", [0, 1, 2, 0, 1, 2, 3]),
    ("ABABABAB", [0, 0, 1, 2, 3, 4, 5, 6]),
    ("AABAACAABAA", [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]),
    ("ACABACACD", [0, 0, 1, 0, 1, 2, 3, 2, 0]),
    ("ABACABABC", [0, 0, 1, 0, 1, 2, 3, 2, 0]),
    ("ABACABABA", [0, 0, 1, 0, 1, 2, 3, 2, 3]),
]


@pytest.mark.parametrize(
    "needle, expected",
    test_cases,
)
def test_compute_lps_array(needle, expected):
    actual = compute_lps_array(needle)
    assert actual == expected


@pytest.mark.parametrize(
    "needle, expected",
    test_cases,
)
def test_compute_lps_array_2(needle, expected):
    assert compute_lps_array_2(needle) == expected


@pytest.mark.parametrize(
    "needle, expected",
    test_cases,
)
def test_compute_lps_array_3(needle, expected):
    assert compute_lps_array_3(needle) == expected
