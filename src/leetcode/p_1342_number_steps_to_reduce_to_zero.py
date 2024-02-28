"""
https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/description/

1342. Number of Steps to Reduce a Number to Zero
Easy

Given an integer num, return the number of steps to reduce it to zero.

In one step, if the current number is even, you have to divide it by 2, otherwise,
you have to subtract 1 from it.



Example 1:
Input: num = 14
Output: 6
Explanation:
Step 1) 14 is even; divide by 2 and obtain 7.
Step 2) 7 is odd; subtract 1 and obtain 6.
Step 3) 6 is even; divide by 2 and obtain 3.
Step 4) 3 is odd; subtract 1 and obtain 2.
Step 5) 2 is even; divide by 2 and obtain 1.
Step 6) 1 is odd; subtract 1 and obtain 0.

Example 2:
Input: num = 8
Output: 4
Explanation:
Step 1) 8 is even; divide by 2 and obtain 4.
Step 2) 4 is even; divide by 2 and obtain 2.
Step 3) 2 is even; divide by 2 and obtain 1.
Step 4) 1 is odd; subtract 1 and obtain 0.

Example 3:
Input: num = 123
Output: 12


Constraints:
0 <= num <= 106
"""


def number_of_steps(num: int) -> int:
    steps = 0
    while num > 0:
        num = num / 2 if num % 2 == 0 else num - 1
        steps += 1
    return steps


def number_of_steps_bitwise(num: int) -> int:
    steps = 0

    while num > 0:
        if num & 1 == 0:
            # num & bitmask: xxxx xxx0 & 0000 0001: 0 --> even number
            # num & bitmask: xxxx xxx1 & 0000 0001: 1 --> odd number
            num >>= 1  # half the value of num
        else:
            num -= 1
        steps += 1
    return steps


if __name__ == "__main__":
    print(number_of_steps(14))  # expect 6
    print(number_of_steps(8))  # expect 4
    print(number_of_steps(123))  # expect 12

    print(number_of_steps_bitwise(14))
    print(number_of_steps_bitwise(8))
    print(number_of_steps_bitwise(123))
