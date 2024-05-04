"""
https://leetcode.com/problems/boats-to-save-people/description/

881. Boats to Save People
Medium

You are given an array people where people[i] is the weight of the ith person,
and an infinite number of boats where each boat can carry a maximum weight of limit.
Each boat carries at most two people at the same time, provided the sum
of the weight of those people is at most limit.

Return the minimum number of boats to carry every given person.



Example 1:
Input: people = [1,2], limit = 3
Output: 1
Explanation: 1 boat (1, 2)

Example 2:
Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)

Example 3:
Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)


Constraints:

1 <= people.length <= 5 * 10**4
1 <= people[i] <= limit <= 3 * 10**4
"""

from typing import List


def num_rescue_boats__inplace(people: List[int], limit: int) -> int:
    people.sort()
    boats, lo, hi = 0, 0, len(people) - 1
    while lo <= hi:
        if people[lo] + people[hi] <= limit:
            lo += 1  # discount if the boat can carry a light person together
        hi -= 1  # always transport the heaviest person
        boats += 1

    return boats


def num_rescue_boats(people: List[int], limit: int) -> int:
    weights = sorted(people)
    boats, lo, hi = 0, 0, len(weights) - 1
    while lo <= hi:
        if weights[lo] + weights[hi] <= limit:
            lo += 1  # discount if the boat can carry a light person together
        hi -= 1  # always transport the heaviest person
        boats += 1

    return boats


if __name__ == "__main__":
    print(num_rescue_boats([3, 2, 2, 1], 3))
    print(num_rescue_boats([2, 1], 3))
    print(num_rescue_boats([3, 5, 3, 4], 5))
    print(num_rescue_boats([2, 2], 6))
