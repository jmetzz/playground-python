"""These algorithms, while primarily discussed in the context of string matching,
are not limited to string matching only. They can be applied to a broader range
of pattern matching problems, including but not limited to:

Searching in Binary Data: These algorithms can be applied to binary data streams where
you're looking for a specific sequence of bytes. This is common in file scanning,
network packet analysis, and other applications where binary data needs to be inspected.

DNA Sequence Analysis: In bioinformatics, sequences of nucleotides in DNA and RNA,
or sequences of amino acids in proteins, can be searched using these algorithms
to find matches or motifs, which are essential for understanding genetic information
and functions.

Plagiarism Detection: Rabin-Karp, in particular, is well-suited for detecting plagiarism
in documents by looking for matching sequences of words or tokens across documents.
This application extends beyond simple character matching to analyzing sequences
of text or code.

Natural Language Processing (NLP): These algorithms can be used to search for patterns in text,
such as finding names, locations, or specific terms in large corpora of text, which
is a common task in NLP applications.

Database Search: Searching for patterns within large databases can utilize these algorithms,
especially when looking for partial matches or substrings within database fields.

Computer Vision: Pattern matching algorithms can be adapted for use in image processing
and computer vision, searching for visual patterns or features within images.
"""

from typing import List


def compute_lps_array(pattern: str) -> List[int]:
    """Computes the Longest Prefix Suffix (LPS) array used in KMP algorithm.

    The LPS array stores important information about how the pattern
    matches agains itself.

    The LPS array helps in pattern searching by telling us the longest matching prefix
    and suffix within a pattern. For every position i in the pattern, the LPS array
    stores the length of the longest prefix which is also a suffix for the substring
    ending at i, except for the entire string.


    Let's consider the pattern "ABABACA" and construct its LPS array step by step:
    Pattern ABABACA
        > Remember, when matching the prefixes and suffixes, do not count the entire substring.

    Step 1: For A (index 0), no proper prefix or suffix that is different
            from the entire substring A, so LPS[0] = 0.
            LPS[0] is always 0.
        A
        LPS: 0
    Step 2: For AB (index 1), no matching prefix and suffix, so LPS[1] = 0.
            The first char A is != the last char B.
            prefixes: A, AB
            suffixese: B, AB
            AB is equal the entire substring, and thus, is disregarded
            prefix A is not equal to suffix B, thus, LPS[1] = 0
        A B
        LPS: 0 0

    Step 3: For ABA (index 2), A is both a prefix and suffix, so LPS[2] = 1.
            prefixes: A, AB, ABA
            suffixese: A, BA, ABA
            ABA is equal the entire substring, and thus, is disregarded
            prefix A is equal suffix A, thus, LPS[2] = 1
        A B A
        LPS: 0 0 1

    Step 4: For ABAB (index 3), AB matches at both ends, so LPS[3] = 2.
        A B A B
        LPS: 0 0 1 2

    Step 5: For ABABA (index 4), ABA matches at both ends, so LPS[4] = 3.
        A B A B A
        LPS: 0 0 1 2 3

    Step 6: For ABABAC (index 5), no matching prefix and suffix, so LPS[5] = 0.
        A B A B A C
        LPS: 0 0 1 2 3 0

    Step 7: For ABABACA (index 6), A matches at both ends, so LPS[6] = 1.
        A B A B A C A
        LPS: 0 0 1 2 3 0 1

    Final LPS Array: 0 0 1 2 3 0 1

    Two good examples to understand the LPS algorith are: AAAXAAAX and AAACAAAA.

    Args:
    ----
        needle (str): The pattern string for which LPS array is computed.

    Returns:
    -------
        List[int]: The LPS array.

    """
    lps = [0] * len(pattern)
    prev_lps = 0  # Length of the previous longest prefix suffix
    i = 1  # the loop starts from 1 since the first character's LPS is always 0
    while i < len(pattern):
        print(f"i: {i}, prev_lps: {prev_lps}, needle[i]: {pattern[i]}, needle[prev_lps]: {pattern[prev_lps]}")
        if pattern[i] == pattern[prev_lps]:
            # Current characters match, increment length of the current LPS
            lps[i] = prev_lps + 1
            prev_lps += 1
            i += 1
        elif prev_lps == 0:
            # No matching prefix found that could be extended,
            # so LPS is 0 for this character
            lps[i] = 0
            i += 1  # advance the pointer to check on the next character
        else:
            # Current characters don't match, meanign the current longest prefix
            # (that is also a suffix) cannot be extended with the character at i.
            # Therefore, fall back to the last "length" where there was a match.
            # This does not change the current index `i` in the pattern,
            # but adjusts `length` to the last known matching prefix-suffix length.
            # Consequently, if in the next iteration needle[i] and needle[prev_lps]
            # still don't match, we will fallback again. This is repeated util we
            # either find a match or go back to start, essentially trying to find
            # the next longest prefix that is also a suffix.
            prev_lps = lps[prev_lps - 1]

        print(f"\t>>> lps: {lps}")
    return lps


def compute_lps_array_2(pattern: str) -> List[int]:
    lps = [0] * len(pattern)
    prev_lps = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[prev_lps]:
            prev_lps += 1
            lps[i] = prev_lps
            i += 1
        elif prev_lps == 0:
            lps[i] = 0
            i += 1
        else:
            prev_lps = lps[prev_lps - 1]

    return lps


def compute_lps_array_3(pattern: str) -> List[int]:
    # Longest Proper Prefix that is suffix array
    lps = [0] * len(pattern)

    prev_lps = 0
    for i in range(1, len(pattern)):
        # Phase 3: roll the prefix pointer back until match or
        # beginning of pattern is reached
        while prev_lps and pattern[i] != pattern[prev_lps]:
            prev_lps = lps[prev_lps - 1]

        # Phase 2: if match, record the LSP for the current `i`
        # and move prefix pointer
        if pattern[prev_lps] == pattern[i]:
            prev_lps += 1
            lps[i] = prev_lps

        # Phase 1: is implicit here because of the for loop and
        # conditions considered above

    return lps


def kmp_search(haystack: str, needle: str) -> int:
    """Searches for the needle in the haystack using the Knuth-Morris-Pratt algorithm.

    KMP is particularly efficient when:
    - The pattern has repeating subpatterns.
    - The same pattern is searched in multiple texts.
     -Avoiding redundant searches in the text is critical for performance.

    Time Complexity: O(n + m), where n is the length of the text and m
    is the length of the pattern. The preprocessing of the pattern
    (LPS array computation) takes O(m), and the actual search takes O(n).

    Space Complexity: O(m) for storing the LPS array.

    Good explanation: https://www.youtube.com/watch?v=JoF0Z7nVSrA

    Args:
    ----
        haystack (str): The text string to search within.
        needle (str): The pattern string to search for.

    Returns:
    -------
        int: The starting index of the first occurrence of needle in haystack,
             or -1 if needle is not present.

    """
    if not needle:
        return 0

    lps = compute_lps_array(needle)
    i = j = 0
    while i < len(haystack):
        if needle[j] == haystack[i]:
            i += 1
            j += 1
        if j == len(needle):
            return i - j
        if i < len(haystack) and needle[j] != haystack[i]:
            if j != 0:
                # Use the LPS array to skip characters avoiding redundant comparisons.
                # When a mismatch happens, the LPS array tells KMP how much of
                # the prefix matches the suffix so far, allowing the algorithm to
                # continue from the longest matched prefix suffix.
                j = lps[j - 1]
            else:
                i += 1
    return -1


def boyer_moore_search(haystack: str, needle: str) -> int:
    """Searches for the needle in the haystack using the Boyer-Moore algorithm.

    This simplified version focuses on the Bad Character Rule to skip sections
    of the text, making it more efficient than brute-force for large texts.

    Args:
    ----
        haystack (str): The text string to search within.
        needle (str): The pattern string to search for.

    Returns:
    -------
        int: The starting index of the first occurrence of needle in haystack,
             or -1 if needle is not present.

    """
    if not needle:
        return 0

    def bad_char_heuristic(needle: str) -> dict:
        bad_char = {}
        for i in range(len(needle)):
            bad_char[needle[i]] = i
        return bad_char

    bad_char = bad_char_heuristic(needle)
    shift = 0
    while shift <= len(haystack) - len(needle):
        j = len(needle) - 1
        while j >= 0 and needle[j] == haystack[shift + j]:
            j -= 1
        if j < 0:
            return shift
        shift += max(1, j - bad_char.get(haystack[shift + j], -1))
    return -1


def rabin_karp_search(haystack: str, needle: str) -> int:
    """Searches for the needle in the haystack using the Rabin-Karp algorithm.

    The algorithm uses hashing to find the pattern in the text. It's particularly
    effective for multiple pattern search scenarios.

    Args:
    ----
        haystack (str): The text string to search within.
        needle (str): The pattern string to search for.

    Returns:
    -------
        int: The starting index of the first occurrence of needle in haystack,
             or -1 if needle is not present.

    """
    if not needle:
        return 0

    d = 256  # Number of characters in the input alphabet
    q = 101  # A prime number
    h = pow(d, len(needle) - 1) % q
    p = t = 0
    for i in range(len(needle)):
        p = (d * p + ord(needle[i])) % q
        t = (d * t + ord(haystack[i])) % q
    for i in range(len(haystack) - len(needle) + 1):
        if p == t and haystack[i : i + len(needle)] == needle:
            return i
        if i < len(haystack) - len(needle):
            t = (d * (t - ord(haystack[i]) * h) + ord(haystack[i + len(needle)])) % q
    return -1
