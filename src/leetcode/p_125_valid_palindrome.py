"""https://leetcode.com/problems/valid-palindrome/description/

125. Valid Palindrome
Easy

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and
removing all non-alphanumeric characters, it reads the same forward and backward.
Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:
Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.


Constraints:
1 <= s.length <= 2 * 105
s consists only of printable ASCII characters.
"""


def is_sentence_palindrome_1(sentence: str) -> bool:
    """The more didactic version for better understanding"""
    # Convert to lowercase and keep alphanumeric characters
    cleaned = "".join(c.lower() for c in sentence if c.isalnum())

    # find the mid-point of the cleaned string
    n = len(cleaned)
    mid = n // 2

    # check if the cleaned string is a palindrome
    # by comparing the two halves.
    # the left half should be equal to the inverted right half
    return cleaned[:mid] == (cleaned[mid:] if n % 2 == 0 else cleaned[mid + 1 :])[::-1]


def is_sentence_palindrome_2(sentence: str) -> bool:
    """The more didactic version for better understanding"""
    # Convert to lowercase and keep alphanumeric characters
    cleaned = "".join(c.lower() for c in sentence if c.isalnum())

    # check if the cleaned string is a palindrome
    # in a more direct approach:
    #   a palindrome string is the same as its inverted string
    return cleaned == cleaned[::-1]


def is_sentence_palindrome_3(sentence: str) -> bool:
    """Faster and lower memory consumption version

    For a more performant solution, we'll still clean the string,
    but aim to reduce memory usage and improve execution speed by
    avoiding the creation of a new cleaned string and
    checking the palindrome in place.

    The rationale is to use two pointers that move in direction of
    each other and are not allowed to cross each other.
    While iterating we find the next alphanumeric character
    in each direction and compare them in a case-sensitive manner.
    If they are different we stop the loop by returning False.
    The, return True if me manage to compare all alphanumeric
    characters in the sentence.

    The same strategy could be implemented with expanding window,
    starting from alphanumeric character in the middle of the sentence and
    moving the pointers outward while making the appropriate comparison.
    However, this strategy adds to the complexity that we need to find
    the first valid character that is in the middle of the sentence.
    """
    left, right = 0, len(sentence) - 1
    while left < right:
        # Move left index to the next alphanumeric character
        while left < right and not sentence[left].isalnum():
            left += 1
        # Move right index to the previous alphanumeric character
        while left < right and not sentence[right].isalnum():
            right -= 1

        # Compare characters in a case-insensitive manner
        if sentence[left].lower() != sentence[right].lower():
            return False

        left, right = left + 1, right - 1
    return True


if __name__ == "__main__":
    sentences = [("A man, a plan, a canal: Panama", True), ("race a car", False), (" ", True)]
    for text, expected in sentences:
        print(f"{text} [expected: {expected}] --> {is_sentence_palindrome_1(text)}")
        print(f"{text} [expected: {expected}] --> {is_sentence_palindrome_2(text)}")
        print(f"{text} [expected: {expected}] --> {is_sentence_palindrome_3(text)}")
