"""
https://leetcode.com/problems/generate-parentheses/description/
22. Generate Parentheses
Medium

Given n pairs of parentheses, write a function to generate all combinations
of well-formed parentheses.

Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:
Input: n = 1
Output: ["()"]

Constraints:
1 <= n <= 8
"""


from typing import List


def generate_parenthesis_1(n: int) -> List[str]:
    target_length = n * 2

    def _backtracking(target_size, diff, comb):
        if len(comb) > target_length or not (0 <= diff <= n):
            # generated more closing parentheses than openning ones
            return
        if target_size == 0 and diff == 0:
            # early stopping condition due to target size.
            # if the current combination is valid (ie, diff == 0),
            # add it to the final list of combinations
            combinations.append(comb)
        else:
            _backtracking(target_size - 1, diff + 1, comb + "(")
            _backtracking(target_size - 1, diff - 1, comb + ")")

    combinations = []
    _backtracking(n * 2, 0, "")
    return combinations


def generate_parenthesis_2(n: int) -> List[str]:
    combinations = []
    target_length = n * 2

    def _backtracking(open_cnt: int, close_cnt: int, comb: str):
        if len(comb) == target_length:
            combinations.append(comb)
            return
        if open_cnt < n:
            _backtracking(open_cnt + 1, close_cnt, comb + "(")
        if close_cnt < open_cnt:
            _backtracking(open_cnt, close_cnt + 1, comb + ")")

    _backtracking(0, 0, "")
    return combinations


def generate_parenthesis_3(n: int) -> List[str]:
    combinations = []

    def _backtracking(open_available: int, close_available: int, comb: str):
        if open_available == 0 and close_available == 0:
            combinations.append(comb)
            return
        if open_available > 0:
            _backtracking(open_available - 1, close_available, comb + "(")
        if close_available > open_available:
            _backtracking(open_available, close_available - 1, comb + ")")

    _backtracking(n, n, "")
    return combinations


def is_valid_comb(comb: str) -> bool:
    diff = 0
    for ch in comb:
        if ch == "(":
            diff += 1
        elif diff == 0:
            return False
        else:
            diff -= 1
    return diff == 0


if __name__ == "__main__":
    print(is_valid_comb("()"))
    print(is_valid_comb("()("))
    print(is_valid_comb("())"))
    print(is_valid_comb("()()"))
    print(is_valid_comb("()(()"))
    print(is_valid_comb("(()())"))
    print(is_valid_comb("(()())("))

    print(generate_parenthesis_1(3))  # expect: [((())), (()()), (())(), ()(()), ()())]
    print(generate_parenthesis_2(3))  # expect: [((())), (()()), (())(), ()(()), ()())]
    print(generate_parenthesis_3(3))  # expect: [((())), (()()), (())(), ()(()), ()())]
