"""
https://leetcode.com/problems/valid-parentheses/description

20. Valid Parentheses
Easy
Topics
Companies
Hint
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.


Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false


Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.
"""


def is_valid_parenthesis(s: str) -> bool:
    pair_of = {")": "(", "]": "[", "}": "{"}
    stack = []

    for ch in s:
        if ch in ("(", "[", "{"):
            stack.append(ch)
        else:
            if not stack or stack[-1] != pair_of[ch]:
                return False
            stack.pop()
    return len(stack) == 0


if __name__ == "__main__":
    print(is_valid_parenthesis("()"))  # true
    print(is_valid_parenthesis("()[]{}"))  # true
    print(is_valid_parenthesis("([()])"))  # true
    print(is_valid_parenthesis("([({}{}[()])])"))  # true

    print(is_valid_parenthesis("(]"))  # false
    print(is_valid_parenthesis("([)]"))  # false
    print(is_valid_parenthesis("([)]"))  # false
    print(is_valid_parenthesis("([()]"))  # false
    print(is_valid_parenthesis("]"))  # false
