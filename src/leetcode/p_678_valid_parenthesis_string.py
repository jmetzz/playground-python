"""
https://leetcode.com/problems/valid-parenthesis-string/description

678. Valid Parenthesis String
Medium

Given a string s containing only three types of characters: '(', ')' and '*',
return true if s is valid.

The following rules define a valid string:

Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left
parenthesis '(' or an empty string "".


Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "(*)"
Output: true

Example 3:
Input: s = "(*))"
Output: true


Constraints:

1 <= s.length <= 100
s[i] is '(', ')' or '*'.

"""


def check_valid_string_recursive(s: str) -> bool:
    """

    The recursive approach will result in Time Limit Exceeded (TLE) issues due
    to the exponential nature of possibilities (3^100 is a huge number).
    """

    def helper(idx, left_counter) -> bool:
        if idx == len(s):
            return left_counter == 0
        if left_counter < 0:
            return False

        match s[idx]:
            case "(":
                return helper(idx + 1, left_counter + 1)
            case ")":
                return helper(idx + 1, left_counter - 1)
            case "*":
                return (
                    helper(idx + 1, left_counter)  # * is treated as ""
                    or helper(idx + 1, left_counter + 1)  # * is treated as "("
                    or helper(idx + 1, left_counter - 1)  # * is treated as ")" --> like the left ( was popped
                )

    return helper(0, 0)


def check_valid_string_two_stacks(s: str) -> bool:
    """Using two stacks

    The first stack keeps track of the indices of encountered open brackets,
    while the second stack is dedicated to storing the indices of asterisks.
    """

    left_p = []
    stars = []
    for idx, c in enumerate(s):
        match c:
            case "*":
                stars.append(idx)
            case "(":
                left_p.append(idx)
            case ")":
                # first attempt to balance this right bracket with an open bracket
                if left_p:
                    # balanced :)
                    left_p.pop()
                elif stars:
                    # no matching left parentheses, try to use a *
                    # to balance if there is any * available
                    stars.pop()
                else:
                    return False

    # We still need to check the remaining elements in the open bracket and
    # asterisk stacks. We need to ensure their relative positions satisfy
    # the constrainsts.
    # If an open bracket appears after the last encountered asterisk,
    # there's no viable way to balance it because we have no available
    # right brackets. If no such mismatch is detected, we proceed to
    # empty both stacks.
    while left_p and stars:
        if left_p.pop() > stars.pop():
            # open bracket appears after the asterisk --> cannot be balanced
            return False
    return not left_p
