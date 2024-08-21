import pytest

from leetcode.p_31_next_permutation import next_permutation


@pytest.mark.parametrize(
    "input_arr, expected", [([1, 2, 3], [1, 3, 2]), ([3, 2, 1], [1, 2, 3]), ([1, 1, 5], [1, 5, 1]), ([1, 2], [2, 1])]
)
def test_next_permutation(input_arr, expected):
    # the function under tests perform in_place operation
    next_permutation(input_arr)
    assert input_arr == expected


def next_permutation_review(nums: list[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    # Observations:
    # 1. a decreasing subsequence (from left to right) cannot have a next lexicographical sorted permutation
    # 2. if an item interrupts the decreasing sequence, it is a pivot we can use to swap with
    #    another element that would make the the next lexicographical order.
    #    The target element is the one that is just larger than this element.
    # 3. Since we interested in the next permutation (in the lexicographical order)
    #    and the subsequence to the right of the element identified in step 1
    #    is in lexicographical order, we can explore the array from right to left
    #    to find the target element to swap.

    # sequence: [1, 5, 8, 4, 7, 6, 5, 3, 1]
    #                        |-----> decreasing subsequence
    #                     ^
    #                     |--- the element just before the decreasing subsequence

    def _reverse_inplace(start):
        left, right = start, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    n = len(nums)
    left = n - 2

    # find the first element that breaks the decreasing sequence
    while left >= 0 and nums[left] >= nums[left + 1]:
        left -= 1

    # [1, 5, 8, 4, 7, 6, 5, 3, 1]
    #           L

    if left > 0:
        # find the "just larger" than nums[left] element to the right of left
        right = n - 1
        while nums[left] >= nums[right]:
            right -= 1
        # [1, 5, 8, 4, 7, 6, 5, 3, 1]
        #           L        R

        # swap places
        nums[left], nums[right] = nums[right], nums[left]
        # [1, 5, 8, 5, 7, 6, 4, 3, 1]
        #           L        R

        # reverse the elements from left + 1
        _reverse_inplace(left + 1)


@pytest.mark.parametrize(
    "input_arr, expected",
    [
        # ([1, 2, 3], [1, 3, 2]), ([3, 2, 1], [1, 2, 3]), ([1, 1, 5], [1, 5, 1]), ([1, 2], [2, 1])
        ([1, 5, 8, 4, 7, 6, 5, 3, 1], [1, 5, 8, 5, 1, 3, 4, 6, 7])
    ],
)
def test_next_permutation_review(input_arr, expected):
    # the function under tests perform in_place operation
    # input_arr = [1, 5, 8, 4, 7, 6, 5, 3, 1]
    next_permutation_review(input_arr)
    assert input_arr == expected
