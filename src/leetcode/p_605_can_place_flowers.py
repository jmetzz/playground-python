"""
https://leetcode.com/problems/can-place-flowers/description/


605. Can Place Flowers
Easy

You have a long flowerbed in which some of the plots are planted, and some are not.
However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and
1 means not empty, and an integer n, return true if n new flowers can be
planted in the flowerbed without violating the no-adjacent-flowers rule
and false otherwise.



Example 1:
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true

Example 2:
Input: flowerbed = [1,0,0,0,1], n = 2
Output: false


Constraints:

1 <= flowerbed.length <= 2 * 104
flowerbed[i] is 0 or 1.
There are no two adjacent flowers in flowerbed.
0 <= n <= flowerbed.length
"""

from typing import List


def can_place_flowers_naive(flowerbed: List[int], n: int) -> bool:
    size = len(flowerbed)
    planted = 0
    for i, v in enumerate(flowerbed):
        if v == 0:
            empty_left = (i == 0) or (flowerbed[i - 1] == 0)
            empty_right = (i == size - 1) or (flowerbed[i + 1] == 0)
            if empty_left and empty_right:
                flowerbed[i] = 1
                planted += 1
                if planted == n:
                    return True

    return planted >= n


def can_place_flowers_padding(flowerbed: List[int], n: int) -> bool:
    bed = [0] + flowerbed + [0]
    for i in range(1, len(flowerbed) + 1):
        if bed[i - 1] == bed[i] == bed[i + 1] == 0:
            bed[i] = 1
            n -= 1
        if n <= 0:
            return True
    return False
