import pytest

from leetcode.p_165_compare_version_numbers import compare_version_1, compare_version_2, compare_version_3


@pytest.mark.parametrize("func", [compare_version_1, compare_version_2, compare_version_3])
@pytest.mark.parametrize(
    "version_1, version_2, expected",
    [
        ("1.01", "1.001", 0),
        ("1.0", "1.0.0", 0),
        ("0.1", "1.1", -1),
        ("1.1", "0.1", 1),
    ],
)
def test_compare_versions(func, version_1, version_2, expected):
    assert func(version_1, version_2) == expected
