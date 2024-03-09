from typing import List


def max_area(height: List[int]) -> int:
    """Two pointers solution

    This approach starts from the widest container
    (left pointer pointing to the left most position and
    right pointer pointing to the right most position)
    and progress by "shrinking the containers" by moving
    the pointer that points to the lower wall inwards.

    Keep in mind that increasing the width of the container
    will increase the volume only if the height of the
    new boundary is greater than before.

    """
    left, right = 0, len(height) - 1
    max_height = max(height)
    _max = float("-inf")
    while left < right:
        _max = max(_max, (right - left) * min(height[left], height[right]))
        # optimizatino step:
        if _max >= max_height * (right - left):
            # no other contaier will have a bigger area
            break
        # adjust the pointers
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return _max
