"""https://leetcode.com/problems/subarray-product-less-than-k/description/

713. Subarray Product Less Than K
Medium

Given an array of integers nums and an integer k, return the number of contiguous subarrays
where the product of all the elements in the subarray is strictly less than k.



Example 1:

Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
Example 2:

Input: nums = [1,2,3], k = 0
Output: 0


Constraints:

1 <= nums.length <= 3 * 104
1 <= nums[i] <= 1000
0 <= k <= 106
"""

from typing import List


def num_subarray_product_less_than_k(nums: List[int], k: int) -> int:
    """Calculates the number of contiguous subarrays where the product of all the elements in the
    subarray is less than a given threshold k.

    This function uses a two-pointer technique to maintain a sliding window of elements. It expands
    and shrinks the window dynamically based on the product of the elements within the window. The
    goal is to include as many contiguous subarrays as possible where their product is less than k.

    Args:
    ----
        nums (List[int]): A list of positive integers representing the given array.
        k (int): The product threshold. The function counts subarrays whose product is less than this value.

    Returns:
    -------
        int: The total number of contiguous subarrays satisfying the product condition.

    Approach:
        - Initialize a counter to keep track of the number of valid subarrays.
        - Use two pointers (lo and hi) to maintain a sliding window. The variable 'window_prod'
          holds the product of the elements within this window.
        - Iterate through the array with the 'hi' pointer, expanding the window by multiplying the
          current element 'e' to 'window_prod'.
        - If 'window_prod' becomes equal to or greater than 'k', shrink the window from the left by
          dividing 'window_prod' by the element at 'lo' and then incrementing 'lo'.
        - After adjusting the window to ensure 'window_prod' is less than 'k', calculate the number
          of valid subarrays that can be formed with the current window. This number is the difference
          between 'hi' and 'lo', plus one (since 'hi - lo + 1' counts the number of elements in the window).
        - Sum these counts to find the total number of subarrays with a product less than 'k'.

    Note:
    ----
        The function returns 0 if 'k' is less than or equal to 1, as the product of any non-empty
        subarray of positive integers is always 1 or more.

    """
    if k <= 1:
        return 0
    window_prod = 1
    counter = 0
    lo = 0
    for hi, e in enumerate(nums):  # controls windows expansion
        # enumerating nums guarantee we only consider each element once
        window_prod *= e  # expand window by including e

        # shrink the window
        while window_prod >= k:
            # discount the element that just dropped out of the window
            window_prod //= nums[lo]
            lo += 1
        # Update the counter:
        # add the number of subarrays withint the current window
        counter += hi - lo + 1
    return counter


def num_subarray_product_less_than_k__double_loop(nums: List[int], k: int) -> int:
    if k < 2:
        return 0

    size = len(nums)
    count = 0
    for i in range(size):
        product = nums[i]

        if product < k:
            count += 1

        for j in range(i + 1, size):
            product *= nums[j]

            if product < k:
                count += 1
            else:
                break

    return count
