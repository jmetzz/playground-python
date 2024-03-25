"""
https://leetcode.com/problems/find-the-duplicate-number/description

287. Find the Duplicate Number
Medium

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.



Example 1:
Input: nums = [1,3,4,2,2]
Output: 2

Example 2:
Input: nums = [3,1,3,4,2]
Output: 3

Example 3:
Input: nums = [3,3,3,3,3]
Output: 3


Constraints:

1 <= n <= 105
nums.length == n + 1
1 <= nums[i] <= n
All the integers in nums appear only once except for precisely one integer which appears two or more times.


Follow up:

How can we prove that at least one duplicate number must exist in nums?
Can you solve the problem in linear runtime complexity?
"""
from collections import Counter, defaultdict
from typing import List


def find_duplicate_brute_force(nums: List[int]) -> int:
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] == nums[j]:
                return nums[i]
    return -1


def find_duplicate_sorting(nums: List[int]) -> int:
    sorted_nums = sorted(nums)
    for i in range(len(nums) - 1):
        if sorted_nums[i] == sorted_nums[i + 1]:
            return sorted_nums[i]
    return -1


def find_duplicate_bool_array(nums: List[int]) -> int:
    n = len(nums)
    seen = [False] * (n + 1)
    for e in nums:
        if seen[e]:
            return e
        seen[e] = True
    return -1


def find_duplicate_hashset(nums: List[int]) -> int:
    seen = set()
    for e in nums:
        if e in seen:
            return e
        seen.add(e)
    return -1


def find_duplicate_builtin_counter(nums: List[int]) -> int:
    counter = Counter(nums)
    answer = [e for e, count in counter.items() if count > 1]
    return answer[0] if len(answer) > 0 else -1


def find_duplicate_dict_counter(nums: List[int]) -> int:
    counter = defaultdict(int)
    for e in nums:
        counter[e] += 1
        if counter[e] > 1:
            return e
    return -1


def find_duplicate_subtract(nums: List[int]) -> int:
    """
    Finds the first duplicate number in the given list of integers by marking visited indices.
    This version avoids modifying the original list by working on a copy, thereby removing side effects.

    The function iterates over the numbers, using their absolute values as indices. If the value at the indexed
    position in the copied list is negative, it indicates the index has been visited before, and thus the corresponding
    number is a duplicate.

    Args:
        nums (List[int]): A list of integers where each integer is between 1 and n, inclusive,
                          with at least one duplicate.

    Returns:
        int: The first duplicate number found in the list. If no duplicate is found, returns -1.

    Examples:
        >>> find_duplicate_subtract([3, 1, 3, 4, 2])
        3
        >>> find_duplicate_subtract([1, 3, 4, 2, 2])
        2
        >>> find_duplicate_subtract([1, 2, 3, 4, 5])
        -1

    Note:
        This implementation does not modify the original input list, making it side-effect-free.
    """
    # Create a copy of nums to avoid modifying the original list
    nums_copy = nums[:]
    for i in range(len(nums_copy)):
        index = abs(nums_copy[i]) - 1
        if nums_copy[index] < 0:
            return index + 1
        nums_copy[index] = -nums_copy[index]
    return -1


def find_duplicate_fast_slow_pointers(nums: List[int]) -> int:
    """Finds the duplicate number in an array containing n + 1 integers

    where each integer is between 1 and n (inclusive).

    Utilizes the Floyd's Tortoise and Hare algorithm to identify the duplicate,
    which is is based on cycle detection in linked lists.

    Mapping the Problem:
    Imagine each position in the array as a node in a linked list,
    and the value at each position as a pointer to the next node.
    This setup creates a virtual linked list where the problem
    of finding a duplicate number becomes analogous to
    finding the start of a cycle within this list.

    Given the problem constraints (n + 1 integers between 1 and n),
    at least one number must be repeated, creating at least one "loop"
    in the virtual linked list. This is because there are more nodes
    than unique possible values (pointers).

    The algorithm consists of two phases:
    1. Fast and slow pointers advance through the array
    (tortoise moves one step at a time, hare moves two steps)
    until they meet within a cycle.
    2. Use a second slow pointer starting the the first position.
    Then move both slow pointeres at the same pace.
    The point at which they meet again is the entry point to the cycle,
    which corresponds to the duplicate number.

    Args:
        nums: A List of integers where each integer is between
        1 and n (inclusive) and one integer appears at least twice.

    Returns:
        The duplicate integer found in the nums list.

    Examples:
        >>> find_duplicate_fast_slow_pointers([1, 3, 4, 2, 2])
        2
        >>> find_duplicate_fast_slow_pointers([3, 1, 3, 4, 2])
        3
    """
    # Phase 1: advance fast and slow pointers until they reach the same element
    # Be mindful we need to start at nums[0] instead of 0,
    # since this aligns with the problem constraints:
    #   treating each value as a pointer to the next index, and
    #   each nums[i] is in [1, n] (inclusive interval).
    fast, slow = nums[0], nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: Find the "entrance" to the cycle
    slow_2 = nums[0]
    while slow != slow_2:
        slow = nums[slow]
        slow_2 = nums[slow_2]

    return slow


def find_duplicate_bitwise(nums: List[int]) -> int:
    """
    Finds the first duplicate number in the given list of integers using bitwise operations.

    This function iterates through the list of numbers, using a bitwise value set to
    keep track of which numbers have been seen. Each integer in the list is used to
    set a corresponding bit in the value set. If a bit corresponding to a number
    is already set when encountered, this number is the first duplicate and is returned.

    Note: This approach assumes that no value in `nums` exceeds the number of bits
    in the integer type used for `value_set`. Given Python's handling of large integers dynamically,
    this is typically not a constraint unless dealing with extremely large numbers.

    Args:
        nums (List[int]): A list of integers where each integer is between 1 and n (inclusive),
        with at least one duplicate.

    Returns:
        int: The first duplicate number found in the list. If no duplicate is found, returns -1.

    Examples:
        >>> find_duplicate_bitwise([3, 1, 3, 4, 2])
        3
        >>> find_duplicate_bitwise([1, 4, 4, 2, 4])
        4
        >>> find_duplicate_bitwise([1, 2, 3, 4, 5])
        -1
    """
    value_set = 0
    for value in nums:
        value_mask = 1 << value
        if value_mask & value_set:
            # The value has already been seen
            return value

        # Add the value to the seen value set
        value_set |= value_mask
    return -1
