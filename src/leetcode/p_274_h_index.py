"""https://leetcode.com/problems/h-index/description/

solution: https://www.youtube.com/watch?v=FvnTWDKT_ck

274. H-Index
Medium

Given an array of integers citations where citations[i] is the number of citations
a researcher received for their ith paper, return the researcher's h-index.

According to the definition of h-index on Wikipedia: The h-index is defined as
the maximum value of h such that the given researcher has published at least
 h papers that have each been cited at least h times.


Example 1:
Input: citations = [3,0,6,1,5]
Output: 3
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and
each of them had received 3, 0, 6, 1, 5 citations respectively.
Since the researcher has 3 papers with at least 3 citations each and
the remaining two with no more than 3 citations each, their h-index is 3.

Example 2:
Input: citations = [1,3,1]
Output: 1


Constraints:

n == citations.length
1 <= n <= 5000
0 <= citations[i] <= 1000
"""


def h_index_sorting_ascending(citations: list[int]) -> int:
    """Use sorting and iteration to find the h-index

    Intuition:
    Consider the input array [3, 0, 6, 1, 5].

    - original:                             [3, 0, 6, 1, 5]
    - sorted:                               [0, 1, 3, 5, 6]

    From this we can find the number of papers that have been
    cited at least m times, with m as the number of citations
    in each paper.
    - sorted:                               [0, 1, 3, 5, 6]
    - h-index candidates:                    5  4  3  4  5
        * papers with (at least) sorted[i] citations

    Then, iterating over the sorted array to identify
    the first element for which the number of real citations
    is equal of greater than the h-index candidate
    """
    citations.sort()
    n = len(citations)
    for idx, num_citations in enumerate(citations):
        if (n - idx) <= num_citations:
            return n - idx
    return 0


def h_index_sorting_descending(citations: list[int]) -> int:
    citations.sort(reverse=True)
    h_index = 0
    for idx, num_citations in enumerate(citations):
        if num_citations >= idx + 1:
            h_index = idx + 1
        else:
            break

    return h_index


def h_index_counting(citations: list[int]) -> int:
    """Use extra space to run in O(n) Time Complexity

    Intuition:
    Consider the input array [3, 0, 6, 1, 5].
    We can use an extra array to count the number of citations
    for each possible h-index position.
    Let's say this array is named frequencies.
    For this, we also need to account for a possible 0 h-index,
    and for every possible paper with more than n citations,
    we accumulate in the last position of the frequencies array.

    We start with zero in all elements of

    - original:                  [3, 0, 6, 1, 5]
    - frequencies:               [1, 1, 0, 1, 0, 2]
        1 paper with zero citation (eg original[1])
        1 paper with 1 citation (eg original[3])
        0 papers with 2 citations
        and so on

    Now we can iterate backwards on the frequencies array
    accumulating the total frequencies (think of a prefix sum array
    from right to left) until we get to the index that is smaller
    or equal to the accumulated citations.
    """
    citations.sort()
    n = len(citations)
    frequencies = [0] * (n + 1)
    for c in citations:
        frequencies[c if c < n else -1] += 1

    # calculate the h-index: iterate backwards
    # accumulating the total number of citations
    # up to each index i
    counter = 0
    for idx in reversed(range(n + 1)):
        counter += frequencies[idx]
        if counter >= idx:
            return idx


if __name__ == "__main__":
    print(h_index_sorting_ascending([3, 0, 6, 1, 5]))
    print(h_index_sorting_descending([3, 0, 6, 1, 5]))
    print(h_index_counting([3, 0, 6, 1, 5]))
