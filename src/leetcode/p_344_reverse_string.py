"""

https://leetcode.com/problems/reverse-string/description/?envType=list&envId=p18vogoj

344. Reverse String
Easy

Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

Example 1:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Example 2:
Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]


Constraints:
1 <= s.length <= 105
s[i] is a printable ascii character.
"""
from typing import List


def reverse_inplace(arr: List[str]) -> None:
    """
    Do not return anything, modify s in-place instead.
    """
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1


if __name__ == "__main__":
    inputs = [
        (["h", "e", "l", "l", "o"], ["o", "l", "l", "e", "h"]),
        (["H", "a", "n", "n", "a", "h"], ["h", "a", "n", "n", "a", "H"]),
        (["h", "e", "l", "l", "o"], ["o", "l", "l", "e", "h"]),
        (list(range(10)), list(reversed(range(10)))),
        (list(range(7)), list(reversed(range(7)))),
    ]
    for input_list, _ in inputs:
        reverse_inplace(input_list)
        print(f"Reversed: {input_list}")
        print(f"Expected: {input_list}")
        print()
