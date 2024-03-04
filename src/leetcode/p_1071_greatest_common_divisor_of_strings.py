"""
https://leetcode.com/problems/greatest-common-divisor-of-strings/description

1071. Greatest Common Divisor of Strings
Easy

For two strings s and t, we say "t divides s" if and only if s = t + t + t + ... + t + t
(i.e., t is concatenated with itself one or more times).

Given two strings str1 and str2, return the largest string x such that x divides both str1 and str2.


Example 1:
Input: str1 = "ABCABC", str2 = "ABC"
Output: "ABC"

Example 2:
Input: str1 = "ABABAB", str2 = "ABAB"
Output: "AB"

Example 3:
Input: str1 = "LEET", str2 = "CODE"
Output: ""


Constraints:
1 <= str1.length, str2.length <= 1000
str1 and str2 consist of English uppercase letters.
"""


def gcd_of_strings(str1: str, str2: str) -> str:
    # we only need to try string lengths up to
    # the length of the shorter of our two input strings.
    # so, s1 represents the shorter string
    s1, s2 = str1, str2
    n1, n2 = len(s1), len(s2)
    if n1 > n2:
        s1, s2 = s2, s1
        n1, n2 = n2, n1

    # start from the longest prefix possible
    for prefix_size in range(n1, 0, -1):
        candidate = s1[:prefix_size]
        # the candidate prefix string needs to evenly divide s1
        # and str2. Thus, we only need to consider prefixes
        # with a length that is evenly divisible into
        # the length of both s1 and s2.
        if n1 % prefix_size != 0 or n2 % prefix_size != 0:
            continue

        s1_multiplier = n1 // prefix_size
        s2_multiplier = n2 // prefix_size

        if s1 == candidate * s1_multiplier and s2 == candidate * s2_multiplier:
            return candidate

    return ""


if __name__ == "__main__":
    inputs = [
        ("ABCABC", "ABC"),
        ("ABABAB", "ABAB"),
        ("LEET", "CODE"),
    ]

    for v, w in inputs:
        print(f">>{gcd_of_strings(v, w)}<<")
