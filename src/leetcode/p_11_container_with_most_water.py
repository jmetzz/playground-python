"""https://leetcode.com/problems/container-with-most-water/description/

11. Container With Most Water
Medium

You are given an integer array height of length n.
There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.



Example 1:
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case,
the max area of water (blue section) the container can contain is 49.

Example 2:
Input: height = [1,1]
Output: 1

Constraints:

n == height.length
2 <= n <= 105
0 <= height[i] <= 104
"""


def max_area(height: list[int]) -> int:
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
