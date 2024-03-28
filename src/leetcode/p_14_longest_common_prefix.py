"""
https://leetcode.com/problems/longest-common-prefix/description/

14. Longest Common Prefix
Easy

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".


Example 1:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.


Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters.
"""

from typing import List


def longest_common_prefix__brute_force(strs: List[str]) -> str:
    if len(strs) == 0:
        return ""

    sorted_strs = sorted(strs, key=lambda s: len(s))
    prefix = sorted_strs[0]
    for i, letter in enumerate(prefix):
        for j in range(1, len(strs)):
            if sorted_strs[j][i] != letter:
                return prefix[:i]
    return prefix


def longest_common_prefix__vertical_scannning(strs: List[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for i, letter in enumerate(prefix):
        for j in range(1, len(strs)):
            if i == len(strs[j]) or strs[j][i] != letter:
                return prefix[:i]
    return prefix


def longest_common_prefix__horizontal_scanning(strs: List[str]) -> str:
    if len(strs) == 0:
        return ""
    prefix = strs[0]
    for other in strs:
        # checks whether prefix is not at the beginning of the other string
        while other.find(prefix) != 0:  # prefix not found at the start of other
            prefix = prefix[:-1]  # shorten prefix by one char
            if not prefix:  # but if prefix becomes empty, return ""
                return ""
    return prefix


def longest_common_prefix__zip_vertical(strs: List[str]) -> str:
    """A clever strategy to find the longest common prefix
    among an array of strings.

    Here's a breakdown of how it works:

    The Use of zip(*strs):
    zip when used with the unpacking operator * on strs essentially combines
    the nth character of each string in strs into a tuple. For example,
    if strs is ["flower", "flow", "flight"], zip will give you:
        ('f', 'f', 'f')
        ('l', 'l', 'l')
        ('o', 'o', 'i')
        ('w', 'w', 'g')
    In the first iteration of the loop, ('f', 'f', 'f') it the char_tuple,
    int the second ('l', 'l', 'l'), and so on.

    Using this tuple we can check if all chars are equal and add to the prefix,
    otherwise, we can stop and return the current prefix.

    This strategy effectively scanns the strings vertically
    (character by character at the same position across all strings)
    until it reaches the end of the shortest string, since zip stops
    when the shortest iterable is exhausted.
    """
    prefix = ""
    for char_tuple in zip(*strs):
        if len(set(char_tuple)) == 1:
            # all chars are the same, then add it to the prefix
            prefix += char_tuple[0]
        else:
            # the current char do not match in all words
            break
    return prefix


def longest_common_prefix__first_and_last(strs: List[str]) -> str:
    """
    Comparing Only the First and Last Words

    After sorting, the function takes the first and the last strings in the sorted list.
    This is based on the insight that if the first and last strings have a common prefix,
    all strings in between must share that prefix.
    This is because the sorting guarantees that any common prefix in the array will be
    shared across the entire range from the first to the last string.

    Example:
    >>> strs = ["flower", "flow", "flight"]
    >>> sorted(strs)
        ['flight', 'flow', 'flower']

    Thus:
    'fl | ight'
    'fl | ow'
    'fl | ower'


    """
    prefix = ""

    sorted_strs = sorted(strs)
    first_word = sorted_strs[0]
    last_word = sorted_strs[-1]

    for i in range(min(len(first_word), len(last_word))):
        if first_word[i] == last_word[i]:
            prefix += first_word[i]
        else:
            return prefix
    return prefix


def longest_common_prefix_trie(strs: List[str]) -> str:
    """
    Find the deepest path in the trie from the root, which satisfies the following conditions:

    - each node along the path must contain only one child element.
    Otherwise the found path will not be a common prefix among all strings.
    - the path doesn't comprise of nodes which are marked as end of key.
    Otherwise the path couldn't be a prefix a of key which is shorter than itself.

    To find the longest common prefix, traverse the Trie form the root until it is
    impossible to continue and satisfy the constraints above at the same time.
    """

    def _trie_insert(key: str) -> None:
        node = trie
        for letter in key:
            if letter not in node:
                node[letter] = {}
            node = node[letter]
        node[EOW] = None

    def _find_longest_common_prefix() -> str:
        prefix = ""
        node = trie
        while node and EOW not in node and len(node) == 1:
            # get the first (and presumably only) key-value pair from the current node
            letter, node = next(iter(node.items()))  # node becomes the next node
            prefix += letter
        return prefix

    if len(strs) == 0:
        return ""

    trie, EOW = {}, "*"
    for word in strs:
        _trie_insert(word)

    return _find_longest_common_prefix()
