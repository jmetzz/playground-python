"""
https://leetcode.com/problems/maximum-odd-binary-number/description

2864. Maximum Odd Binary Number
Easy

You are given a binary sequence s that contains at least one '1'.

You have to rearrange the bits in such a way that the resulting binary number is
the maximum odd binary number that can be created from this combination.

Return a sequence representing the maximum odd binary number that can be created
from the given combination.

Note that the resulting sequence can have leading zeros.


Example 1:
Input: s = "010"
Output: "001"
Explanation: Because there is just one '1', it must be in the last position. So the answer is "001".

Example 2:
Input: s = "0101"
Output: "1001"
Explanation: One of the '1's must be in the last position. The maximum number that can be made with
the remaining digits is "100". So the answer is "1001".


Constraints:
1 <= s.length <= 100
s consists only of '0' and '1'.
s contains at least one '1'.

"""
from collections import Counter


def maximum_odd_binary_number_1(sequence: str) -> str:
    counter = Counter(sequence)
    return "1" * (counter["1"] - 1) + "0" * counter["0"] + "1"


def maximum_odd_binary_number_2(sequence: str) -> str:
    ones, zeros = 0, 0
    for e in sequence:
        if e == "1":
            ones += 1
        else:
            zeros += 1
    return "1" * (ones - 1) + "0" * zeros + "1"


def maximum_odd_binary_number_3(sequence: str) -> str:
    return "".join(sorted(sequence)[::-1][1:] + ["1"])


def maximum_odd_binary_number_4(sequence: str) -> str:
    ones = sequence.count("1")
    zeros = len(sequence) - ones
    answer = ""
    for _ in range(1, ones):
        answer += "1"
    for _ in range(zeros):
        answer += "0"
    return answer + "1"


def maximum_odd_binary_number_5(sequence: str) -> str:
    sorted_s = sorted(sequence, reverse=True)
    for i in reversed(range(len(s) - 1)):
        # find the first 1 from right to left
        if sorted_s[i] == "1":
            # swap with the first last element which is a zero
            sorted_s[i], sorted_s[-1] = sorted_s[-1], sorted_s[i]
            break
    return "".join(sorted_s)


def maximum_odd_binary_number_6(sequence: str) -> str:
    sorted_s = sorted(sequence)
    idx = sorted_s.index("1")
    sorted_s[idx], sorted_s[0] = sorted_s[0], sorted_s[idx]
    return "".join(sorted_s[::-1])


if __name__ == "__main__":
    inputs = ["010", "0101", "010111", "01000010000"]
    for s in inputs:
        print(maximum_odd_binary_number_1(s))
        print(maximum_odd_binary_number_2(s))
        print(maximum_odd_binary_number_3(s))
        print(maximum_odd_binary_number_4(s))
        print(maximum_odd_binary_number_5(s))
        print(maximum_odd_binary_number_6(s))
        print("----")
