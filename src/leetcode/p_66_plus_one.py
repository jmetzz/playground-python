"""https://leetcode.com/problems/plus-one/description/

66. Plus One

You are given a large integer represented as an integer array digits,
where each digits[i] is the ith digit of the integer.
The digits are ordered from most significant to least significant
in left-to-right order. The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.



Example 1:
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].

Example 2:
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
Incrementing by one gives 4321 + 1 = 4322.
Thus, the result should be [4,3,2,2].

Example 3:
Input: digits = [9]
Output: [1,0]
Explanation: The array represents the integer 9.
Incrementing by one gives 9 + 1 = 10.
Thus, the result should be [1,0].


Constraints:
1 <= digits.length <= 100
0 <= digits[i] <= 9
digits does not contain any leading 0's.
"""

from typing import List


def plus_one(digits: List[int]) -> List[int]:
    answer = [0] * len(digits)
    reminder = 1
    for i in range(len(digits) - 1, -1, -1):
        d = reminder + digits[i]
        answer[i] = d if d <= 9 else 0
        reminder = d // 10
    return answer if reminder == 0 else [1] + answer


def plus_one_early_stop(digits: List[int]) -> List[int]:
    answer = [0] * len(digits)

    for i in range(len(digits) - 1, -1, -1):
        if digits[i] == 9:
            answer[i] = 0
        else:
            answer[i] = digits[i] + 1
            # early stopping
            return digits[:i] + answer[i:]
    return [1] + answer


def plus_one_early_stop_inplace(digits: List[int]) -> List[int]:
    """Change the input array and return a reference to it."""
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] == 9:
            digits[i] = 0
        else:
            digits[i] += 1
            return digits
    digits.insert(0, 1)
    return digits


if __name__ == "__main__":
    inputs = [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 9], [9], [9, 9]]
    for arr in inputs:
        print(plus_one(arr))
        print(plus_one_early_stop(arr))
        print(plus_one_early_stop_inplace(arr))
        print()

    # check for side effects
    arr = [9, 9]
    print(f"input: {arr} --> result: {plus_one(arr)} --> arr after call funtion: {arr}")
    print(f"input: {arr} --> result: {plus_one_early_stop(arr)} --> arr after call funtion: {arr}")
