"""https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description

80. Remove Duplicates from Sorted Array II
Medium

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place
such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must
instead have the result be placed in the first part of the array nums. More formally,
if there are k elements after removing the duplicates, then the first k elements of
nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array.
You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

```
int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
```

If all assertions pass, then your solution will be accepted.


Example 1:
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Explanation: Your function should return k = 5, with the first five elements of
nums being 1, 1, 2, 2 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
Explanation: Your function should return k = 7, with the first seven elements of
nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).


Constraints:

1 <= nums.length <= 3 * 104
-104 <= nums[i] <= 104
nums is sorted in non-decreasing order.

"""


def remove_duplicates(nums: list[int]) -> int:
    """Removes duplicates from the sorted array such that each element appears at most twice.
    This function modifies the input array in place and returns the new length of the array.

    Approach:
    - Use a two-pointer technique:
        - `slow`: The slow pointer tracks the position where the next unique or
                  allowed duplicate element should be placed.
        - `fast`: The fast pointer iterates through the array to find the next element to process.

    - The array is sorted, so duplicates will be consecutive.
    - The first two elements are always allowed because they can appear at most twice.
    - For each element starting from the third one (index 2), compare it with the element
        located two positions behind (`nums[slow - 2]`):
        - If they are different, this means the current element can be added to the
          position indicated by the `slow` pointer.
        - If they are the same, skip the current element as adding it would exceed the
          allowed count of two.

    Args:
    ----
        nums (list[int]): A list of integers sorted in non-decreasing order.

    Returns:
    -------
        int: The length of the modified array where each element appears at most twice.

    Example:
    -------
        >>> nums = [0, 0, 1, 1, 1, 1, 2, 3, 3]
        >>> length = removeDuplicates(nums)
        >>> print(length)  # Output: 7
        >>> print(nums[:length])  # Output: [0, 0, 1, 1, 2, 3, 3]

    """
    # If the array has 2 or fewer elements, it's already valid; no modification needed.
    if len(nums) <= 2:
        return len(nums)

    # Initialize the slow pointer at index 2, since the first two elements are allowed.
    slow = 2

    # Iterate through the array starting from the third element.
    for fast in range(2, len(nums)):
        # Compare the current element with the element two positions before the slow pointer.
        # If they are different, copy the current element to the slow pointer's position
        # and move the slow pointer forward.
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1

    return slow


def remove_duplicates_2(nums: list[int]) -> int:
    if len(nums) <= 2:
        return len(nums)

    slow, fast = 0, 0
    n = len(nums)
    while fast < n:
        # iterates through the array using the fast pointer.
        count = 1
        while fast + 1 < n and nums[fast] == nums[fast + 1]:
            # counts how many times the current number repeats consecutively.
            fast += 1
            count += 1

        # Copying Elements:
        # ensures that at most two occurrences of the current number
        # are copied to the slow pointer's position in the array.
        for _ in range(min(2, count)):
            # pointer moves forward for each copy,
            # ensuring that the array is modified in place.
            nums[slow] = nums[fast]
            slow += 1

        # incremented the fast pointer to start processing
        # the next number or sequence.
        fast += 1

    # the slow pointer indicates the new length of the array
    return slow


if __name__ == "__main__":
    input_arr = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    length = remove_duplicates(input_arr)
    print(length)  # Output: 7
    print(input_arr[:length])  # Output: [0,0,1,1,2,3,3]

    input_arr = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    length = remove_duplicates_2(input_arr)
    print(length)  # Output: 7
    print(input_arr[:length])  # Output: [0,0,1,1,2,3,3]

    input_arr = [1, 2, 2]
    length = remove_duplicates(input_arr)
    print(length)  # Output: 2
    print(input_arr[:length])
