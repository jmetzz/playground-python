"""
https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses

1614. Maximum Nesting Depth of the Parentheses
Easy

A string is a valid parentheses string (denoted VPS) if it meets one of the following:

It is an empty string "", or a single character not equal to "(" or ")",
It can be written as AB (A concatenated with B), where A and B are VPS's, or
It can be written as (A), where A is a VPS.
We can similarly define the nesting depth depth(S) of any VPS S as follows:

depth("") = 0
depth(C) = 0, where C is a string with a single character not equal to "(" or ")".
depth(A + B) = max(depth(A), depth(B)), where A and B are VPS's.
depth("(" + A + ")") = 1 + depth(A), where A is a VPS.
For example, "", "()()", and "()(()())" are VPS's (with nesting depths 0, 1, and 2),
and ")(" and "(()" are not VPS's.

Given a VPS represented as string s, return the nesting depth of s.


Example 1:
Input: s = "(1+(2*3)+((8)/4))+1"
Output: 3
Explanation: Digit 8 is inside of 3 nested parentheses in the string.

Example 2:
Input: s = "(1)+((2))+(((3)))"
Output: 3


Constraints:

1 <= s.length <= 100
s consists of digits 0-9 and characters '+', '-', '*', '/', '(', and ')'.
It is guaranteed that parentheses expression s is a VPS.
"""

from itertools import accumulate


def max_depth_cheeting(expr: str) -> int:
    return max(accumulate(map(lambda c: 1 if c == "(" else -1, filter(lambda c: c in "()", expr))), default=0)


def max_depth_counting(expr: str) -> int:
    max_depth = 0
    counter = 0
    for i in expr:
        if i == "(":
            counter += 1
        elif i == ")":
            counter -= 1

        max_depth = max(max_depth, counter)
    return max_depth


def max_depth_stack(expr: str) -> int:
    """Assumes the input expr is a VPS expression

    VPS = valid parentheses string
    """
    stack = []
    max_depth = 0
    for c in expr:
        if c == "(":
            stack.append(c)
        elif c == ")":
            stack.pop()
            curr_depth = len(stack) + 1
            if curr_depth > max_depth:
                max_depth = curr_depth
    return max_depth


def max_depth_unrestricted(expr: str) -> int:
    """Does not assume the input expr is a VPS expression

    VPS = valid parentheses string
    """
    stack = []
    max_depth = 0
    for c in expr:
        if c == "(":
            stack.append(c)
        elif c == ")":
            if not stack:
                return 0  # invalid VPS
            else:
                stack.pop()
                curr_depth = len(stack) + 1
                if curr_depth > max_depth:
                    max_depth = curr_depth
    return max_depth if not stack else 0
