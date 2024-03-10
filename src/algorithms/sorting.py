import random
import timeit
from ast import Tuple
from random import randint
from typing import List


def merge_sort(elements: List[int]) -> List[int]:
    def _merge_sort_recursive(lo, hi) -> None:
        if lo < hi:
            m = (lo + hi) // 2
            _merge_sort_recursive(lo, m)
            _merge_sort_recursive(m + 1, hi)
            _merge(lo, m, hi)

    def _merge(lo, mid, hi):
        temp = [0] * (hi - lo + 1)
        left = lo
        right = mid + 1
        target_idx = 0

        # while both sides of the array have elements
        # we move them in order to temp
        while left <= mid and right <= hi:
            if elements[left] < elements[right]:
                temp[target_idx] = elements[left]
                left += 1
            else:
                temp[target_idx] = elements[right]
                right += 1
            target_idx += 1
        # add the remaining elements from left subarray to temp
        while left <= mid:
            temp[target_idx] = elements[left]
            left += 1
            target_idx += 1
        # add the remaining elements from right subarray to temp
        while right <= hi:
            temp[target_idx] = elements[right]
            right += 1
            target_idx += 1
        # copy over the elements from temp to the original array
        for i in range(lo, hi + 1):
            elements[i] = temp[i - lo]

    _merge_sort_recursive(0, len(elements) - 1)

    return elements


def quick_sort_lomuto(elements: List[int]) -> List[int]:
    def _lomuto_recursive(lo, hi):
        if lo < hi:
            pivot = _lomuto_partition(elements, lo, hi)
            # at this point the pivot element is in the correct position,
            # and we only need to sort the left and right subarrays
            _lomuto_recursive(lo, pivot - 1)
            _lomuto_recursive(pivot + 1, hi)

    _lomuto_recursive(0, len(elements) - 1)
    return elements


def _lomuto_partition(elements: List[int], lo, hi) -> int:
    pivot = elements[hi]
    # temporary pivot target idx
    swap_marker = lo

    # find the correct position for the pivot element
    for curr in range(lo, hi):
        if elements[curr] < pivot:
            # current element is already (correctly)
            # at the left of the pivot.
            # Float it further back and advanve the
            # temporary target pivot position 1 step forward
            elements[swap_marker], elements[curr] = elements[curr], elements[swap_marker]
            swap_marker += 1

    # move the pivot element to the correct pivot position,
    # between the smaller and larger values
    # current elements from 0..pswap_idx (inclusive) are smaller than pivot
    # and elemetns from pswap_idx..(hi-1) (inclusive) are bigger than pivot.
    # Thus, we only need to swap the pivot with the first element of the right subarray
    elements[hi], elements[swap_marker] = elements[swap_marker], elements[hi]
    # return new position (index) of the pivot
    return swap_marker


def quick_sort_hoare(elements: List[int]) -> List[int]:
    def _hoare_partition(lo, hi):
        pivot = elements[lo]
        left = lo - 1
        right = hi + 1
        while True:
            left += 1
            while elements[left] < pivot:
                left += 1
            right -= 1
            while elements[right] > pivot:
                right -= 1
            if left >= right:
                return right
            # swap:
            elements[left], elements[right] = elements[right], elements[left]

    def _hoare_recursive(lo, hi):
        if lo < hi:
            p = _hoare_partition(lo, hi)
            _hoare_recursive(lo, p)
            _hoare_recursive(p + 1, hi)

    _hoare_recursive(0, len(elements) - 1)
    return elements


def quick_sort_random_pivot(elements: List[int]) -> List[int]:
    def _random_pivot_partition(lo, hi) -> int:
        pivot_idx = random.randint(lo, hi)
        # Move pivot to end for simplicity
        elements[pivot_idx], elements[hi] = elements[hi], elements[pivot_idx]
        # now we can follow the same logic as lomuto:
        return _lomuto_partition(elements, lo, hi)

    def _recursive_call(lo, hi):
        if lo < hi:
            pivot = _random_pivot_partition(lo, hi)
            # at this point the pivot element is in the correct position,
            # and we only need to sort the left and right subarrays
            _recursive_call(lo, pivot - 1)
            _recursive_call(pivot + 1, hi)

    _recursive_call(0, len(elements) - 1)
    return elements


def quick_sort_median_of_three(elements: List[int]) -> List[int]:
    def _recursive_call(lo: int, hi: int) -> None:
        if lo < hi:
            pivot_index = _median_of_three(lo, hi)
            # Move pivot to end
            elements[pivot_index], elements[hi] = elements[hi], elements[pivot_index]
            # reuse the lomuto partiotion method
            p = _lomuto_partition(elements, lo, hi)
            _recursive_call(lo, p - 1)
            _recursive_call(p + 1, hi)

    def _median_of_three(lo: int, hi: int) -> int:
        mid = lo + (hi - lo) // 2
        if elements[lo] > elements[hi]:
            elements[lo], elements[hi] = elements[hi], elements[lo]
        if elements[mid] > elements[hi]:
            elements[mid], elements[hi] = elements[hi], elements[mid]
        if elements[lo] < elements[mid]:
            return mid
        return lo

    _recursive_call(0, len(elements) - 1)
    return elements


def quick_sort_three_way(elements: List[int]) -> List[int]:
    """ "
    Three-Way Partitioning Quick Sort (aka Dutch National Flag Problem)
    is designed to efficiently handle arrays with many duplicate elements
    by partitioning the array into three parts:
        less than,
        equal to, and
        greater than the pivot.
    It's especially useful when the dataset contains a lot of repeated elements.
    """

    def _three_way_partition(lo: int, hi: int) -> Tuple:
        pivot = elements[lo]
        lt = lo
        gt = hi
        i = lo
        while i <= gt:
            if elements[i] < pivot:
                elements[lt], elements[i] = elements[i], elements[lt]
                lt += 1
                i += 1
            elif elements[i] > pivot:
                elements[gt], elements[i] = elements[i], elements[gt]
                gt -= 1
            else:
                i += 1
        return lt, gt

    def _recursive_call(lo, hi):
        if lo < hi:
            lt, gt = _three_way_partition(lo, hi)
            _recursive_call(lo, lt - 1)
            _recursive_call(gt + 1, hi)

    _recursive_call(0, len(elements) - 1)
    return elements


def counting_sort(elements: List[int]) -> List[int]:
    """Counting Sort is a non-comparison-based sorting algorithm
    This implementation of the algorithm works with non-negative elements only.

    That operates by counting the number of objects that have each distinct key value.
    It's particularly efficient for sorting lists of integers when the range of
    potential items (the difference between the maximum and minimum elements)
    is not significantly larger than the number of items.
    Unlike comparison-based sorting algorithms like Quick Sort or Merge Sort,
    which have a runtime complexity of O(n.logn) in their average or worst cases,
    Counting Sort has a linear runtime complexity of O(n + k),
    where n is the number of elements in the input array and k is the range of the input.

    How Counting Sort Works
    - Find the Range: Determine the smallest and largest elements in the input array.
    - Count the Elements: Create a count array to store the count of each distinct element.
    Initialize each value to 0, then iterate through the input array, incrementing the
    count value for each element.
    - Accumulate the Counts: Transform the count array by adding the count of the
    previous indices. This step gives you the positions of the elements in the sorted array.
    - Place the Elements: Iterate through the input array, placing each element at
    its correct position in the sorted array, and decrease its count in the count array.

    Implementation Note for Negative Numbers
    Counting Sort typically does not handle negative numbers because it uses the elements
    of the input array as indices in the count array. To sort arrays that include
    negative numbers, modifications are necessary, such as adjusting the range to
    include negative values or using a different sorting algorithm for mixed-sign data.
    """
    if not elements:
        return []

    # find the max_idx element
    # Example: [4, 2, 2, 8, 3, 3, 1]; max_idx = 8
    max_idx = 0
    size = len(elements)
    for index in range(0, size):
        if elements[index] > elements[max_idx]:
            max_idx = index

    # initialize and 0-array of size max_idx + 1, which is used to store
    # the count of the elements present in the input array
    # at it's respective index in the count array.
    # Example: [4, 2, 2, 8, 3, 3, 1] -> [0, 1, 2, 2, 1, 0, 0, 0, 1]
    count = [0] * (elements[max_idx] + 1)
    for index in range(0, size):
        count[elements[index]] += 1

    # Store cumulative sum of the elements of the count array.
    # It helps in placing the elements into the correct index.
    # If there are x elements less than y, its position
    # should be at x-1.
    # For example: In the accumulated sum array [0, 1, 3, 5, 6, 6, 6, 6, 7],
    # the count of 4 is 6. It denotes that there are 5 elements smaller
    # than 4. Thus, the position of 4 in the sorted array is 5th.
    for index in range(1, len(count)):
        count[index] += count[index - 1]

    # Find the index of each element of the original array in count array.
    # This gives the cumulative count.
    # Place the element at the index calculated.
    # After placing each element at its correct position, decrease its count by one.
    # Example:
    # original array: [4, 2, 2, 8, 3, 3, 1]
    #                  *
    #                  |
    #                  |-----------|
    #                              |
    #                              *
    #           count [0, 1, 3, 5, 6, 6, 6, 6, 7]
    #                              *
    #                              |
    #                              ----
    #                                  |
    #                                  *
    #           output [0, 0, 0, 0, 0, 4, 0]
    #    updated count [0, 1, 3, 5, 5, 6, 6, 6, 7]
    #
    # original array: [4, 2, 2, 8, 3, 3, 1]
    #                     *
    #                     |
    #                     |--|
    #                        |
    #                        *
    #           count [0, 1, 3, 5, 5, 6, 6, 6, 7]
    #                        *
    #                        |
    #                        --
    #                         |
    #                         *
    #           output [0, 0, 2, 0, 0, 4, 0]
    #    updated count [0, 1, 2, 5, 5, 6, 6, 6, 7]

    output = [0] * size
    for index in range(0, size):
        e = elements[index]
        element_count = count[e]
        output[element_count - 1] = e
        count[e] -= 1

    # overwrite the input array
    for i in range(0, size):
        elements[i] = output[i]

    return elements


def setup_data(num_items: int = 10, val_range: Tuple = None):
    """Function to set up the data for the partition functions."""
    val_range = val_range or (-(2**31), 2**31 - 1)
    return [randint(*val_range) for _ in range(num_items)]


def measure_partition_perf(array_size: int, num_runs: int, suffix: str = "1") -> float:
    # Measure performance of partition_1
    time_taken = timeit.timeit(
        stmt=f"partition_{suffix}(elements[:], lo, hi)",  # Use a slice of elements to avoid modifying the original list
        setup=f"from __main__ import partition_{suffix}, setup_data; elements, lo, hi = setup_data({array_size})",
        globals=globals(),  # Provide the global namespace so timeit can access the functions and variables
        number=num_runs,
    )
    return time_taken


def measure_sorting_perf(func_name: str, array_size: int, num_runs: int, value_range: Tuple = None) -> float:
    time_taken = timeit.timeit(
        stmt=f"{func_name}(elements[:])",  # Use a slice of elements to avoid modifying the original list
        setup=f"from __main__ import {func_name}, setup_data; elements = setup_data({array_size}, {value_range})",
        globals=globals(),  # Provide the global namespace so timeit can access the functions and variables
        number=num_runs,
    )
    return time_taken


if __name__ == "__main__":
    data = [4, 2, 2, 8, 3, 3, -1]
    print(merge_sort(data[:]))
    print(quick_sort_lomuto(data[:]))
    print(quick_sort_hoare(data[:]))
    print(quick_sort_random_pivot(data[:]))
    print(quick_sort_median_of_three(data[:]))
    print(quick_sort_three_way(data[:]))
    print(counting_sort(data[:]))

    print(setup_data())
    print(">" * 10 + " Check prformance differences:")
    num_elements = 100000
    num_exec = 100

    sorting_functions = (
        merge_sort,
        quick_sort_lomuto,
        quick_sort_hoare,
        quick_sort_random_pivot,
        quick_sort_median_of_three,
        quick_sort_three_way,
    )
    for f in sorting_functions:
        print(f"'{f.__name__}' took " f"{measure_sorting_perf(f.__name__, num_elements, num_exec):.5f} seconds")
