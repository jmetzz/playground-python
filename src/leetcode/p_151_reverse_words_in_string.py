"""https://leetcode.com/problems/reverse-words-in-a-string/description/
151. Reverse Words in a String
Medium

Given an input string s, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that s may contain leading or trailing spaces or multiple spaces between two words.
The returned string should only have a single space separating the words. Do not include any extra spaces.



Example 1:

Input: s = "the sky is blue"
Output: "blue is sky the"
Example 2:

Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
Example 3:

Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.


Constraints:

1 <= s.length <= 104
s contains English letters (upper-case and lower-case), digits, and spaces ' '.
There is at least one word in s.


Follow-up: If the string data type is mutable in your language, can you solve it in-place with O(1) extra space?
"""


def reverse_words_recursive(s: str) -> str:
    def _reverse(sequence: str) -> str:
        if sequence == "":
            return ""
        sequence = sequence.strip()
        idx = sequence.find(" ")
        if idx == -1:
            return sequence

        return _reverse(sequence[idx + 1 :]) + " " + sequence[:idx].lstrip()

    return _reverse(s)


def reverse_words_iterative(s: str) -> str:
    answer = []
    n = len(s)
    left = n - 1
    while left >= 0:
        # from right to left, find the first char != " "
        while left >= 0 and s[left] == " ":
            left -= 1
        if left < 0:
            break
        right = left
        while left >= 0 and s[left] != " ":
            left -= 1
        # found the word boundaries, copy to answer
        answer.append(s[left + 1 : right + 1])

    return " ".join(answer)


def reverse_words_iterative_2(s: str) -> str:
    answer = ""
    n = len(s)
    left = n - 1
    while left >= 0:
        # from right to left, find the first char != " "
        while left >= 0 and s[left] == " ":
            left -= 1
        if left < 0:
            break
        right = left
        while left >= 0 and s[left] != " ":
            left -= 1
        # found the word boundaries, copy to answer
        answer += s[left + 1 : right + 1] + " "
    return answer[:-1]


def reverse_words_using_builtin_split(s: str) -> str:
    answer = [w for w in s.split(" ")[::-1] if w]
    return " ".join(answer)


if __name__ == "__main__":
    inputs = [
        "the sky is blue",
        "  hello world  ",
        "  hello    world  ",
        "a good   example",
        "a",
        "     a",
        "a     ",
        "    a     ",
        "",
        "     ",
    ]
    for ipt in inputs:
        print(f"#{reverse_words_recursive(ipt)}#")
        print(f"#{reverse_words_iterative(ipt)}#")
        print(f"#{reverse_words_iterative_2(ipt)}#")
        print(f"#{reverse_words_using_builtin_split(ipt)}#")
        print()
