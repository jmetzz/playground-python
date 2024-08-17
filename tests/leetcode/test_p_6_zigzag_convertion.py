import pytest
from leetcode.p_6_zigzag_convertion import zigzag_conversion


@pytest.mark.parametrize(
    "input_str, num_rows, expected",
    [
        ("A", 1, "A"),
        ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
        ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
    ],
)
def test_zigzag_conversion(input_str, num_rows, expected):
    assert zigzag_conversion(input_str, num_rows) == expected
