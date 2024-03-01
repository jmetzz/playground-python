"""
https://leetcode.com/problems/longest-substring-without-repeating-characters/description/

3. Longest Substring Without Repeating Characters
Medium

Given a string s, find the length of the longest
substring without repeating characters.

A substring is a contiguous non-empty sequence of characters within a string.


Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.


Constraints:

0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces.
"""


def length_of_longest_substring__fixed_sliding_window(sequence: str) -> int:
    n = len(sequence)
    if n == 0:
        return 0

    win_size = n
    while win_size > 1:
        num_steps = n - win_size + 1
        for step in range(num_steps):
            window = sequence[step : step + win_size]
            if len(set(window)) == win_size:
                return win_size
        win_size -= 1
    # if the executions reached this point
    # only individual characters can be used
    return 1


def length_of_longest_substring__dynamic_sliding_window_1(sequence: str) -> int:
    chars_in_use = set()
    answer = 0
    left = 0
    for right in range(len(sequence)):
        # right point controls the window expansion
        # if we find a duplicate while trying to expand the window,
        # ie, the duplicate char is at index right,
        # we need to shrink the window removing characters from
        # the left until we eliminate the duplicate
        while sequence[right] in chars_in_use:
            chars_in_use.remove(sequence[left])
            left += 1

        # now we are allowed to add the right most character
        # to char_set without introducing repetition
        chars_in_use.add(sequence[right])
        answer = max(answer, right - left + 1)
    return answer


def length_of_longest_substring__dynamic_sliding_window_2(sequence: str) -> int:
    left = result = 0
    chars_in_use = {}
    for right, char in enumerate(sequence):
        if chars_in_use.get(char, -1) >= left:
            left = chars_in_use[char] + 1
        result = max(result, right - left + 1)
        chars_in_use[char] = right
    return result


if __name__ == "__main__":
    inputs = [
        "abcabcbb",  # 3
        "bbbbb",  # 1
        "pwwkew",  # 3
        "dvdf",  # 3
        "   ",  # 1
        " ",  # 1
        "",  # 0
    ]
    for text in inputs:
        print(length_of_longest_substring__fixed_sliding_window(text))
        print(length_of_longest_substring__dynamic_sliding_window_1(text))
        print(length_of_longest_substring__dynamic_sliding_window_2(text))
        print()
