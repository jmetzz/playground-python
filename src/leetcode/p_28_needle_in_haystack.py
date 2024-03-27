"""
https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/

28. Find the Index of the First Occurrence in a String
Easy

Given two strings needle and haystack, return the index of the first occurrence
of needle in haystack, or -1 if needle is not part of haystack.


Example 1:
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.

Example 2:
Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.


Constraints:

1 <= haystack.length, needle.length <= 104
haystack and needle consist of only lowercase English characters.

"""


def str_in_str_bruteforce(haystack: str, needle: str) -> int:
    """
    Searches for the first occurrence of `needle` in `haystack` using a brute-force approach.

    The function iterates over `haystack`, comparing each substring of length equal to `needle`
    with `needle` itself. If a match is found, the function returns the starting index of the
    substring in `haystack`. If `needle` is not found, the function returns -1. An empty `needle`
    is considered found at position 0.

    Args:
        haystack (str): The string to search within.
        needle (str): The string to search for.

    Returns:
        int: The index of the first character of the first occurrence of `needle` in `haystack`,
             or -1 if `needle` is not found.

    The brute-force method checks every possible position of `needle` within `haystack`, making it
    straightforward but potentially inefficient for large strings or strings with many repeating
    patterns.

    Time complexity is O(m*n), where:
        - n is the length of the haystack string,
        - m is the length of the needle string.

    The brute-force approach is simple and doesn't require any preprocessing of the strings,
    but for large strings or more efficient searching, especially when the same needle is
    searched for in multiple haystacks, algorithms like Knuth-Morris-Pratt (KMP), Boyer-Moore,
    or Rabin-Karp are more efficient in terms of time complexity.
    See 'algorithms.strings.matching' module.

    """
    if len(needle) == 0:
        return 0

    needle_size = len(needle)
    for i in range(len(haystack) - needle_size + 1):  # +1 ensures to include the end of the haystack
        # Check if the substring of haystack starting at i matches needle
        if haystack[i : i + needle_size] == needle:
            return i
    return -1


def str_in_str__cheeting(haystack: str, needle: str) -> int:
    return haystack.find(needle)
