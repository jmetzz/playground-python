"""https://leetcode.com/problems/fizz-buzz/description/

412. Fizz Buzz
Easy

Given an integer n, return a string array answer (1-indexed) where:

answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
answer[i] == "Fizz" if i is divisible by 3.
answer[i] == "Buzz" if i is divisible by 5.
answer[i] == i (as a string) if none of the above conditions are true.


Example 1:
Input: n = 3
Output: ["1","2","Fizz"]

Example 2:
Input: n = 5
Output: ["1","2","Fizz","4","Buzz"]

Example 3:
Input: n = 15
Output: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]

Constraints:
1 <= n <= 104
"""

from typing import List


def fizz_buzz_iterative(n: int) -> List[str]:
    answer = []
    for v in range(1, n + 1):
        fb = ""
        if v % 3 == 0:
            fb += "Fizz"
        if v % 5 == 0:
            fb += "Buzz"
        if (v % 3 != 0) and (v % 5 != 0):
            fb = str(v)
        answer.append(fb)
    return answer


def fizz_buzz_iterative_2(n: int) -> List[str]:
    answer = []
    for v in range(1, n + 1):
        fb = ""
        if v % 3 == 0:
            fb += "Fizz"
        if v % 5 == 0:
            fb += "Buzz"
        answer.append(fb if len(fb) > 0 else str(v))
    return answer


def fizz_buzz_ifelse(n: int) -> List[str]:
    answer = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            answer.append("FizzBuzz")
        elif i % 3 == 0:
            answer.append("Fizz")
        elif i % 5 == 0:
            answer.append("Buzz")
        else:
            answer.append(str(i))
    return answer


def fizz_buzz_pattern_matching(n: int) -> List[str]:
    answer = []
    for v in range(1, n + 1):
        match (v % 3, v % 5):
            case (0, 0):
                answer.append("FizzBuzz")
            case (0, _):
                answer.append("Fizz")
            case (_, 0):
                answer.append("Buzz")
            case _:
                answer.append(str(v))
    return answer


if __name__ == "__main__":
    # expected: ["1","2","Fizz"]
    print(fizz_buzz_iterative(3))
    print(fizz_buzz_iterative_2(3))
    print(fizz_buzz_ifelse(3))
    print(fizz_buzz_pattern_matching(3))

    # expected: ["1","2","Fizz","4","Buzz"]
    print(fizz_buzz_iterative(5))
    print(fizz_buzz_iterative_2(5))
    print(fizz_buzz_ifelse(5))
    print(fizz_buzz_pattern_matching(5))

    # expected: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
    print(fizz_buzz_iterative(15))
    print(fizz_buzz_iterative_2(15))
    print(fizz_buzz_ifelse(15))
    print(fizz_buzz_pattern_matching(15))
