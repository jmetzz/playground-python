"""https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description

2444. Count Subarrays With Fixed Bounds
Hard

You are given an integer array nums and two integers minK and maxK.

A fixed-bound subarray of nums is a subarray that satisfies the following conditions:

The minimum value in the subarray is equal to minK.
The maximum value in the subarray is equal to maxK.
Return the number of fixed-bound subarrays.

A subarray is a contiguous part of an array.


Example 1:
Input: nums = [1,3,5,2,7,5], minK = 1, maxK = 5
Output: 2
Explanation: The fixed-bound subarrays are [1,3,5] and [1,3,5,2].

Example 2:
Input: nums = [1,1,1,1], minK = 1, maxK = 1
Output: 10
Explanation: Every subarray of nums is a fixed-bound subarray. There are 10 possible subarrays.


Constraints:

2 <= nums.length <= 105
1 <= nums[i], minK, maxK <= 106

"""


def count_subarrays(nums: list[int], min_k: int, max_k: int, debug=False) -> int:
    """To count the number of new valid subarrays that end at the current index,
    we look at the distance between the current position and the position
    just after the last "bad" element. This tells us how many subarrays
    including the current element are valid.

    We calculate this distance based on the minimum of lo_boundary and hi_boundary
    (whichever of min_k or max_k was seen last) and subtract the index of the
    last bad element. This effectively counts every valid subarray ending here
    that includes both min_k and max_k and excludes any subarray that includes
    a "bad" element:
        `min(lo_boundary, hi_boundary) - bad_element_idx`.

    Example:
    -------
    For the given array [1, 2, 5, 2, 1, 7, 5] with min_k = 1 and max_k = 5,

    when we are at index 2 (idx = 2, num value = 5), the variables are as follows:
    lo_boundary = 0 (index of the last occurrence of minK = 1),
    hi_boundary = 2 (index of the last occurrence of maxK = 5),
    bad_element_idx = -1 (no bad elements encountered so far).
    To count the number of new valid subarrays that end at index 2,
    we calculate the distance from the current position to the position
    after the most recent bad element (or the start of the array if no bad elements):
    min(lo_boundary, hi_boundary) - bad_element_idx = min(0, 2) - (-1) = 0 - (-1) = 1.
    This calculation indicates that there is 1 valid subarray ending at index 2:
    - The subarray [1, 3, 5], which starts from index 0 and ends at index 2.
    - The subarray [3, 5], which starts from index 1 is not valid, since it does not includ 5.

    In the next iteration idx = 3,
    lo_boundary = 0 (no change)
    hi_boundary = 2 (no change),
    bad_element_idx = -1 (still not found)
    Thus, we add one extra subarray to out total_counter:
    min(lo_boundary, hi_boundary) - bad_element_idx = min(0, 2) - (-1) = 0 - (-1) = 1.

    In the next iteration idx = 4, num = 1.
    Note this time we found the lower boundary value again, and thus update its last seen index.
    lo_boundary = 4
    hi_boundary = 2 (no change),
    bad_element_idx = -1 (still not found)
    Also, since we have seen both lo and hi boundaries, we need to calculates the extra subarrays:
    min(lo_boundary, hi_boundary) - bad_element_idx = min(4, 2) - (-1) = 2 - (-1) = 3.
    This accounts for the new subarrays [2,5,2,1], [5,2,1], and [1,2,5,2,1]
    which were not included in the previous iterations.

    In the next iteration idx = 5 and num = 0, things change a bit
    lo_boundary = 4 (no change)
    hi_boundary = 2 (no change),
    bad_element_idx = 5 (we found a 0 which is outside the valid boundaries)
    Thus, we need to "reset the proceedings". The calculation block is not executed.
    If we were to calculate if anyways, it would result in negative number:
    min(lo_boundary, hi_boundary) - bad_element_idx = min(4, 2) - (5) = 2 - 5 = -3.
    which we would discard.

    """
    total_counter = 0
    lo_boundary = hi_boundary = bad_element_idx = -1
    # bad_element_idx tracks the last occurrence (index) of an element
    # that is outside the bounds defined by minK and maxK.
    # This element "breaks" the fixed-bound subarray condition.
    # And thus, we can use this idx to effectively "resets"
    # the search for new fixed-bound subarrays,
    # as any subarray containing this element is invalid.

    for idx, num in enumerate(nums):
        if debug:
            print(f">>> Start: (lo: {lo_boundary}, hi: {hi_boundary}, bad: {bad_element_idx})")
        if num < min_k or num > max_k:
            if debug:
                print(f"\tReset at nums[{idx}] = {num}")
            bad_element_idx = idx
        if num == min_k:
            lo_boundary = idx
        if num == max_k:
            hi_boundary = idx

        if debug:
            print(f"\tpotential increment: {min(lo_boundary, hi_boundary) - bad_element_idx}")
            print(f"<<< End: (lo: {lo_boundary}, hi: {hi_boundary}, bad: {bad_element_idx})")
            print()
        # Only add to the total count of subarrays if we've seen both min_k and max_k
        # after the most recent element that doesn't fit within the [min_k, max_k] range.
        if lo_boundary > bad_element_idx and hi_boundary > bad_element_idx:
            total_counter += min(lo_boundary, hi_boundary) - bad_element_idx

    return total_counter


if __name__ == "__main__":
    test_cases = [
        [1, 1, 1, 1],
        [1, 3, 5, 2, 7, 5],
        [1, 2, 5, 2, 1, 0, 7, 2, 5],
    ]
    print(count_subarrays(test_cases[2], 1, 5, True))
