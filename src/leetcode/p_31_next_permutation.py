"""https://leetcode.com/problems/next-permutation/description/

31. Next Permutation
Medium

A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

For example, for arr = [1,2,3], the following are all the permutations of
arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].

The next permutation of an array of integers is the next lexicographically greater permutation of its integer.
More formally, if all the permutations of the array are sorted in one container according to their
lexicographical order, then the next permutation of that array is the permutation that follows it in
the sorted container. If such arrangement is not possible, the array must be rearranged as the
lowest possible order (i.e., sorted in ascending order).

For example, the next permutation of arr = [1,2,3] is [1,3,2].
Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a
lexicographical larger rearrangement.
Given an array of integers nums, find the next permutation of nums.

The replacement must be in place and use only constant extra memory.


Example 1:
Input: nums = [1,2,3]
Output: [1,3,2]

Example 2:
Input: nums = [3,2,1]
Output: [1,2,3]

Example 3:
Input: nums = [1,1,5]
Output: [1,5,1]


Constraints:
1 <= nums.length <= 100
0 <= nums[i] <= 100
"""


def next_permutation_review(nums: list[int]) -> None:
    """Do not return anything, modify nums in-place instead

    Observations:
    1. a decreasing subsequence (from left to right) cannot have
       a next lexicographical sorted permutation
       Example: in [... 4, 7, 6, 5, 3, 1] from 7 onwards,
       it is not possible to permute elements and make
       the sequence lexicographical bigger
    2. if an item interrupts the decreasing sequence, it is a pivot
       we can use to swap with another element (to the right)
       that would make the the lexicographical order higher.
       Example: in [...8, 4, 7, 6, 5, 3, 1];
       4 and 7 are the pair of elements that breaks the decreasing sequence.
       Then, 4 is our select pivot element.
    3. Since we interested in the next permutation (in the lexicographical order)
       and the subsequence to the right of the pivot element
       is in lexicographical order, we can explore the array
       from right to left to find the target element to swap with
       the pivot. The target element is the one that is just larger
       than the pivot.
        Example:
    sequence: [1, 5, 8, 4, 7, 6, 5, 3, 1]
                           |-----> decreasing subsequence
                        ^
                        |--- pivot: the element just before the decreasing subsequence
    """

    def _reverse_inplace(start):
        left, right = start, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    n = len(nums)
    pivot = n - 2
    # find the first element that breaks the decreasing sequence
    while pivot >= 0 and nums[pivot] >= nums[pivot + 1]:
        pivot -= 1

    # [1, 5, 8, 4, 7, 6, 5, 3, 1]
    #           L

    if pivot >= 0:
        # find the "just larger" than nums[left] element to the right of left
        swap_target = n - 1
        while nums[pivot] >= nums[swap_target]:
            swap_target -= 1
        # [1, 5, 8, 4, 7, 6, 5, 3, 1]
        #           L        R

        # swap places
        nums[pivot], nums[swap_target] = nums[swap_target], nums[pivot]
        # [1, 5, 8, 5, 7, 6, 4, 3, 1]
        #           L        R

    # reverse the elements from left + 1
    _reverse_inplace(pivot + 1)


def next_permutation(nums: list[int], verbose: bool = False) -> None:
    """Do not return anything, modify nums in-place instead.

    any given sequence that is in descending order,
    can not have a next larger permutation.
    Example: [9, 5, 4, 3, 1]
    while [5, 4, 9, 3, 1] has a element that
    breaks the decreasing order (e.g. 9)
    """

    def _reverse_in_place(start):
        left, right = start, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    # Thus, from right to left find the first 'decreasing' element index
    n = len(nums)
    pivot = n - 2
    while pivot >= 0 and nums[pivot] >= nums[pivot + 1]:
        pivot -= 1

    if pivot >= 0:
        # find the next number just larger than nums[left]
        # on the right side of nums[left]
        swap_target = n - 1
        while nums[pivot] >= nums[swap_target]:
            swap_target -= 1
        # swap them
        nums[pivot], nums[swap_target] = nums[swap_target], nums[pivot]

    # reverse the reminder of the array
    _reverse_in_place(start=pivot + 1)

    if verbose:
        print(nums)


if __name__ == "__main__":
    inputs = [
        [1, 2, 3],  # [1, 3, 2]
        [3, 2, 1],  # [1, 2, 3]
        [1, 1, 5],  # [1, 5, 1]
        [1, 2],  # [2,1]
    ]
    functions = [next_permutation, next_permutation_review]
    for input_arr in inputs:
        for func in functions:
            arr = input_arr[:]
            func(arr)
            print(arr)
        print()
