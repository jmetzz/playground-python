"""
https://leetcode.com/problems/sort-characters-by-frequency/description/


451. Sort Characters By Frequency
Medium

Given a string s, sort it in decreasing order based on the frequency of the characters.
The frequency of a character is the number of times it appears in the string.

Return the sorted string. If there are multiple answers, return any of them.

Example 1:
Input: s = "tree"
Output: "eert"
Explanation: 'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

Example 2:
Input: s = "cccaaa"
Output: "aaaccc"
Explanation: Both 'c' and 'a' appear three times, so both "cccaaa" and "aaaccc" are valid answers.
Note that "cacaca" is incorrect, as the same characters must be together.

Example 3:
Input: s = "Aabb"
Output: "bbAa"
Explanation: "bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.


Constraints:
1 <= s.length <= 5 * 105
s consists of uppercase and lowercase English letters and digits.
"""
from collections import Counter
from heapq import heapify, heappop


class FrequencySort:
    def solve_with_counter(self, s: str) -> str:
        """
        Beats 36.99% of users with Python3
        """
        counter = Counter(s)
        answer = ""
        while counter:
            key, qtt = counter.most_common(1)[0]
            answer += key * qtt
            del counter[key]
        return answer

    def solve_with_array(self, s: str) -> str:
        """
        Beats 53.32% of users with Python3
        """
        counter = Counter(s)

        elements = sorted([(qtt, char) for char, qtt in counter.items()], reverse=True)

        answer = ""
        for e in elements:
            answer += e[0] * e[1]

        return answer

    def solve_with_heap(self, s: str) -> str:
        counter = Counter(s)
        min_heap = [(-qtt, sub_sequence * qtt) for sub_sequence, qtt in counter.items()]
        heapify(min_heap)
        answer = ""
        while min_heap:
            _, sub_sequence = heappop(min_heap)
            answer += sub_sequence

        return answer


if __name__ == "__main__":
    print(FrequencySort().solve_with_heap("tree"))  # -> eetr
    print(FrequencySort().solve_with_heap("cccaaa"))  # -> "aaaccc"
    print(FrequencySort().solve_with_heap("Aabb"))  # -> "bbAa"
    print(FrequencySort().solve_with_heap("raaeaedere"))  # -> "eeeeaaarrd"
    print(FrequencySort().solve_with_heap("abaccadeeefaafcc"))  # -> "aaaaacccceeeffdb"
