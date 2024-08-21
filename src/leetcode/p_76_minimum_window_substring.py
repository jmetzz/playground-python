"""https://leetcode.com/problems/minimum-window-substring/description

76. Minimum Window Substring
Hard

Given two strings s and t of lengths m and n respectively, return the minimum window
substring of s such that every character in t (including duplicates) is included in the window.
If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.



Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.


Constraints:

m == s.length
n == t.length
1 <= m, n <= 105
s and t consist of uppercase and lowercase English letters.



Other sliding window problems to practice:
https://leetcode.com/problems/sliding-window-maximum/
https://leetcode.com/problems/minimum-size-subarray-sum/
https://leetcode.com/problems/minimum-window-substring/
https://leetcode.com/problems/maximum-length-of-repeated-subarray/



Hint 1: Use two pointers to create a window of letters in s,
which would have all the characters from t.

Hint 2: Expand the right pointer until all the characters of t are covered.

Hint 3: Once all the characters are covered, move the left pointer
and ensure that all the characters are still covered to minimize the subarray size.

Hint 4: Continue expanding the right and left pointers until
you reach the end of s.


Explanation:

Goal: find the minimum window in string s that contains all the characters in string t, in ANY ORDER.

Technique: Sliding Window
- Use two pointers (l and r) represent the left and right ends of the current window
- Use a map (ie, dict) to count occurrences of characters in string t.
- Use a curr_win_count object to keeps track of the number of characters within the current window.
- To maintain the window, you can either expand or shrink it.
  - To expand, you need to move the right pointer to expand the window, and check if
  the characters in the current windows statisfy the required counts from t. If satisfied,
  increment formed (count of characters satisfying the required counts)
  - To shrink, you need to move the left pointer and update the minimum window size if a smaller
  valid window is found. Continue until a smaller valid window is not possible of the right pointer
  reaches the end of the string
"""

from collections import Counter, defaultdict


def min_window_solve(source: str, target: str) -> str:
    if not source or not target:
        return ""

    t_counter_map = Counter(target)
    required_count = len(t_counter_map)
    num_chars_used_from_target = 0

    # keep the window meta info
    win_size = -1
    win_left = 0
    win_right = 0
    win_counts = defaultdict(int)  # represent the num of occurrences of char in the window {c: int}

    left, right = 0, 0  # window control pointers
    while right < len(source):  # expand the window iterating over all elements in s
        curr_char = source[right]  # the character from s we are evaluating
        win_counts[curr_char] += 1  # while expanding the window, we add 1 for each char we encounter

        if curr_char in t_counter_map and win_counts[curr_char] == t_counter_map[curr_char]:
            num_chars_used_from_target += 1  # add 1 to represent another char is "consumed" from target

        # shrink the window while the window satisfy the constraint: all chars from target consumed
        while left <= right and num_chars_used_from_target == required_count:
            curr_char = source[left]

            if win_size == -1 or right - left + 1 < win_size:
                win_size = right - left + 1  # length of the answer
                win_left = left  # answer start index
                win_right = right  # answer end index

            win_counts[curr_char] -= 1  # while shrinking the window, we subtract 1 for each char we encounter
            if curr_char in t_counter_map and win_counts[curr_char] < t_counter_map[curr_char]:
                # subtracting 1 represents another char is "put back" in the target, thus, not consumed
                num_chars_used_from_target -= 1
            left += 1

        right += 1

    return "" if win_size == -1 else source[win_left : win_right + 1]
