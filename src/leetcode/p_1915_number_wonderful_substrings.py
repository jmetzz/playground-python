"""https://leetcode.com/problems/number-of-wonderful-substrings/description

1915. Number of Wonderful Substrings
Medium
Topics: #Hash-Table #String #Bit-Manipulation #Prefix-Sum


A wonderful string is a string where at most one letter appears an odd number of times.

For example, "ccjjc" and "abb" are wonderful, but "ab" is not.
Given a string word that consists of the first ten lowercase English letters ('a' through 'j'),
return the number of wonderful non-empty substrings in word.
If the same substring appears multiple times in word, then count each occurrence separately.

A substring is a contiguous sequence of characters in a string.


Example 1:
Input: word = "aba"
Output: 4
Explanation: The four wonderful substrings are underlined below:
- "aba" -> "a"
- "aba" -> "b"
- "aba" -> "a"
- "aba" -> "aba"

Example 2:
Input: word = "aabb"
Output: 9
Explanation: The nine wonderful substrings are underlined below:
- "aabb" -> "a"
- "aabb" -> "aa"
- "aabb" -> "aab"
- "aabb" -> "aabb"
- "aabb" -> "a"
- "aabb" -> "abb"
- "aabb" -> "b"
- "aabb" -> "bb"
- "aabb" -> "b"

Example 3:
Input: word = "he"
Output: 2
Explanation: The two wonderful substrings are underlined below:
- "he" -> "h"
- "he" -> "e"


Constraints:

1 <= word.length <= 105
word consists of lowercase English letters from 'a' to 'j'.

Hint 1: For each prefix of the string, check which characters are of even frequency and
which are not and represent it by a bitmask.
Hint 2: Find the other prefixes whose masks differs from the current prefix mask
by at most one bit.


Editorial approach: Count Parity Prefixes

Intuition
There are two types of wonderful strings:
- those with no letters appearing an odd number of times,
- and those with exactly one letter appearing an odd number of times.

After we find a solution to count the first type of strings, we can adapt it to cover all cases.

The `parity` of a letter means whether the count of that letter in a word is even or odd.
We can find the parity of a letter by taking the frequency of that letter modulo 2.
Letters with odd frequencies have a parity of 1, and letters with even frequencies have a parity of 0.
For example, the parity of letter "a" in "abccada" is 1, whereas the parity of letter "c" is 0.

The subtask now is to count the number of substrings with all letters appearing an even number of times.
In other words, substrings where the parity of every letter is 0. Because there are
only 10 distinct letters the string can consist of, we can use a bitmask of 10 bits to represent
the parities of all letters in a string. The 0th bit of the mask corresponds to the parity of letter "a",
the 1st bit corresponds to letter "b", and so on.

For example, the mask corresponding to string "feffaec" is 100101, which equals 37 in base 10.
Letters "a", "c", and "f" appear an odd number of times, so their corresponding bits are set to 1,
and the other letters appear an even number of times, so their bits are set to 0.

We want to count the number of substrings with a mask of 0 (if every character appears an even number of times,
all bits will be set to 0).

For any substring in the input string word, we can represent it as the difference between two prefixes of s.
For example, substring [2,5] is the difference between prefix [0,5] and [0,1].
Observe that the substring will equate to a mask of 0 if and only if the masks of the two prefixes are equal.
This is because we can "subtract" the larger prefix from the smaller prefix to create this substring
using the ^ (XOR) operator. The XOR function is equivalent to subtraction under modulo 2.
All bits are independently calculated in the XOR function, where for each bit, the output is true when
there is an odd number of true inputs. This gives us an efficient way to find the difference between
the larger and smaller prefixes.

This gives us a linear time way to count strings with all characters appearing an even number of times:
maintain the parity mask of the current prefix, and compare it with previous prefixes of the same value
in a frequency map. The key is a mask, which corresponds to a prefix of the string, and the value
is the frequency of the key mask. To count substrings with all even letters ending at some index r,
take the prefix ending at r with parity mask m, and add freq[m] to the answer.
The difference of two prefixes with the same bitmask will equal 0, which corresponds to strings
with all even frequency letters.

All that's left is to account for the case where exactly one letter appears an odd number of times.
For the current prefix mask, we can find its counterpart in the frequency map by iterating through
which bit should be flipped. For example, if the current prefix mask is 111, and a smaller prefix
has mask 101, the substring generated by removing the intersection of these two prefixes
will equal 010, which means only the letter "b" appears an odd number of times.

Algorithm
1. Create a frequency table or map. Add the mask 000 to account for the empty prefix.
2. Initialize a mask int variable to 0.
3. For each character in word, flip the corresponding bit in mask.
4. Add the frequency of mask to the answer.
5. Increment the value associated with key mask by one.
6. Iterate through each possible character that appears an odd number of times,
and add the frequency of mask ^ (1 << odd_c), where ^ is the XOR function.
7. Return the result when all letters are processed.


Complexity Analysis
Time complexity: O(NA).
The number of distinct characters that can appear in word is defined as A.
For each of the N characters in word, we iterate through all possible characters
that can be the odd character. Therefore, the time complexity of O(NA),
where A≤10, because only letters "a" through "j" will appear.

Space complexity: O(N).
The frequency map can store up to N key/entry pairs, hence the linear space complexity.

"""


def wonderful_substrings(word: str) -> int:
    # Create the frequency map
    # {key: value}, where key is bitmask and value is the corresponding frequency
    freq = {}

    # The empty prefix can be the smaller prefix, which is handled like this
    freq[0] = 1

    mask, answer = 0, 0
    for letter in word:
        bits = ord(letter) - 97

        # Flip the parity of the c-th bit in the running prefix mask
        mask ^= 1 << bits

        # Count smaller prefixes that create substrings with no odd occurring letters
        if mask in freq:
            answer += freq[mask]
            freq[mask] += 1
        else:
            freq[mask] = 1

        # Loop through every possible letter that can appear an odd number of times in a substring
        for odd_c in range(10):
            key = mask ^ (1 << odd_c)
            if key in freq:
                answer += freq[key]

    return answer
