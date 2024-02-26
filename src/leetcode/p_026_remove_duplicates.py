from typing import List


def duplicates_remover(nums: List[int]) -> int:
    n = len(nums)
    anchor = 0
    i = 1
    while i < n:
        if nums[i] == nums[anchor]:
            i += 1
        else:
            nums[anchor + 1] = nums[i]
            anchor += 1
    return anchor + 1
