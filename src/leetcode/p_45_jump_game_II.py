"""
https://leetcode.com/problems/jump-game-ii/description

45. Jump Game II
Medium

You are given a 0-indexed array of integers nums of length n.
You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i.
In other words, if you are at nums[i], you can jump to any nums[i + j] where:

0 <= j <= nums[i] and
i + j < n

Return the minimum number of jumps to reach nums[n - 1].
The test cases are generated such that you can reach nums[n - 1].


Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2


Constraints:
1 <= nums.length <= 10^4
0 <= nums[i] <= 1000
It's guaranteed that you can reach nums[n - 1].

explanation: https://www.youtube.com/watch?v=dJ7sWiOoK7g

"""


def jump_greedy(nums: list[int]) -> int:
    """Time complexity O(n)

    bfs like solution.

    From each position in the array (from left to right),
    we create a window of elements that can be reached.
    This means, with 1 jump only, any element in the window
    can be reached from the previous position.

    To update the window, we need to move the left pointer
    to the next element of the right pointer (starting the new
    scope window), while the right pointer is updated with the
    index of the farthest element we can reach from that window.

    Example:
    nums [2, 3, 1, 1, 4]
             [l  r] -> scope window from position zero,
                 one jump to reach any of the elements in the window

    The farthest index we can reach from that window is the index 4 (which is
    the target index btw), since left + nums[l] is 4. Thus, pretty much from any
    position in the window we can reach the position 4 with 1 jump.

    from this scenario, left pointer is updated to index 3 (which is right + 1)
    and the right pointer is updated to the farthest index (eg 4).

    """
    left = right = 0  # window pointers
    jumps = 0
    while right < len(nums) - 1:
        farthest = 0
        for i in range(left, right + 1):
            farthest = max(farthest, i + nums[i])
        left = right + 1
        right = farthest
        jumps += 1
    return jumps


if __name__ == "__main__":
    print(jump_greedy([2, 3, 1, 1, 4]))  # 2
    print(jump_greedy([2, 3, 0, 1, 4]))  # 2
