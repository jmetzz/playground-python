"""41. First Missing Positive
Hard

Given an unsorted integer array nums. Return the smallest positive integer that is not present in nums.

You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.



Example 1:

Input: nums = [1,2,0]
Output: 3
Explanation: The numbers in the range [1,2] are all in the array.
Example 2:

Input: nums = [3,4,-1,1]
Output: 2
Explanation: 1 is in the array but 2 is missing.
Example 3:

Input: nums = [7,8,9,11,12]
Output: 1
Explanation: The smallest positive integer 1 is missing.


Constraints:

1 <= nums.length <= 105
-231 <= nums[i] <= 231 -


Topics: #Array #Hash-Table

Hint 1:
Think about how you would solve the problem in non-constant space.
Can you apply that logic to the existing space?

Hint 2:
We don't care about duplicates or non-positive integers.

Hint 3:
Remember that O(2n) = O(n)

"""

from typing import List


def first_missing_positive_filter(nums: List[int]) -> int:
    positives = set([e for e in nums if e > 0])
    n = len(positives) + 1
    for e in range(1, n + 1):
        if e not in positives:
            return e
    return n + 1


def first_missing_positive_set(nums: List[int]) -> int:
    elements = set(nums)
    missing = 1
    while missing in elements:
        missing += 1
    return missing


def first_missing_positive_boolean_array(nums: List[int]) -> int:
    """Complexity Analysis
    Let n be the length of nums.

    Time complexity: O(n):
        Marking the values from nums in seen takes O(n).
        We check for values 1 to n in seen, which takes O(n).
        The total time complexity will be O(2n), which we can simplify to O(n)O(n)O(n).

    Space complexity: O(n)
        We initialize the array seen, which is size n + 1,
        so the space complexity is O(n).
    """
    n = len(nums)
    seen = [False] * (n + 1)  # Array for lookup

    # Mark the elements from nums in the lookup array
    for num in nums:
        if 0 < num <= n:
            seen[num] = True

    # Iterate through integers 1 to n
    # return smallest missing positive integer
    for i in range(1, n + 1):
        if not seen[i]:
            return i

    # If seen contains all elements 1 to n
    # the smallest missing positive number is n + 1
    return n + 1


def first_missing_positive_side_effect(nums: List[int]) -> int:
    """Time: O(n)
    Space: O(1) since we modify the input array [Don't do this, ever!]
    """
    n = len(nums)
    contains_1 = False

    # Replace negative numbers, zeros, and numbers larger than n with 1s.
    for i in range(n):
        # Check whether 1 is in the original array
        if nums[i] == 1:
            contains_1 = True
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = 1

    if not contains_1:
        return 1

    # Mark whether integers 1 to n are in nums
    # Use index as a hash key and negative sign as a presence detector.
    for i in range(n):
        value = abs(nums[i])
        # If you meet number a in the array - change the sign of a-th element.
        # Be careful with duplicates : do it only once.
        if value == n:
            nums[0] = -abs(nums[0])
        else:
            nums[value] = -abs(nums[value])

    # First positive in nums is smallest missing positive integer
    for i in range(1, n):
        if nums[i] > 0:
            return i

    # nums[0] stores whether n is in nums
    if nums[0] > 0:
        return n

    # If nums contained all elements 1 to n
    # the smallest missing positive number is n + 1
    return n + 1


def first_missing_positive_cycle_sort(nums: List[int]) -> int:
    """https://en.wikipedia.org/wiki/Cycle_sort:
    sort a given sequence in a range from a to n by putting each element at
    the index that corresponds to its value.
    Since nums is a zero-indexed array, an element with the value x will be
    located at index x - 1.

    Then, to determine the smallest positive integer, we iterate through nums,
    and return the first element that is not equal to its index plus one.

    If we iterate through the whole sorted array without returning a value,
    the array consists of the sequence of numbers 1 through n, so we return n + 1.


    """
    n = len(nums)

    # Place positive elements smaller than n at the correct index
    i = 0
    while i < n:
        correct_idx = nums[i] - 1
        if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
            # swap
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # Iterate through nums and return smallest missing positive integer
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # If all elements are at the correct index
    # the smallest missing positive number is n + 1
    return n + 1
