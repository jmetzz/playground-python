"""https://leetcode.com/problems/zigzag-conversion/description

6. Zigzag Conversion
Medium

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number
of rows like this: (you may want to display this pattern in a fixed font for
better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);


Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I
Example 3:

Input: s = "A", numRows = 1
Output: "A"


Constraints:

1 <= s.length <= 1000
s consists of English letters (lower-case and upper-case), ',' and '.'.
1 <= numRows <= 1000
"""

from collections import defaultdict


def zigzag_conversion(s: str, num_rows: int) -> str:
    if num_rows == 1:
        return s

    _map = defaultdict(list)

    key, delta = 0, -1
    for letter in s:
        _map[key].append(letter)
        if key == num_rows - 1 or key == 0:
            delta *= -1
        key += delta
    answer = ["".join(v) for v in _map.values()]
    return "".join(answer)


if __name__ == "__main__":
    print(zigzag_conversion("PAYPALISHIRING", 3))  # PAHNAPLSIIGYIR
    print(zigzag_conversion("PAYPALISHIRING", 4))  # PINALSIGYAHRPI
