"""https://leetcode.com/problems/asteroid-collision/description

735. Asteroid Collision
Medium

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction
(positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode.
If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.


Example 1:
Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

Example 2:
Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.

Example 3:
Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.


Constraints:
2 <= asteroids.length <= 104
-1000 <= asteroids[i] <= 1000
asteroids[i] != 0
"""

from typing import List


def asteroid_collision(asteroids: List[int]) -> List[int]:
    """The key insight for this problem is:
        > The positions in the array represent their relative positions in space,
        > moving either to the left or right. All asteroids are considered to be moving simultaneously.

    This means, asteroids at the left most position in the array and moving left,
    will colide.
    A collision can only occur if there are asteroids in the stack (representing right-moving asteroids
    that have not collided) and the current asteroid a is moving to the left (i.e., a < 0).
    Furthermore, for a collision to happen, the asteroid at the top of the stack must be moving to
    the right (stack[-1] > 0).

    O(n) time complexity, where n is the number of asteroids

    """
    stack = []
    for a in asteroids:
        while stack and a < 0 and stack[-1] > 0:
            # colision
            diff = a + stack[-1]
            if diff < 0:
                stack.pop()  # Incoming asteroid destroys the top of the stack asteroid
            elif diff > 0:
                a = 0  # Top of the stack asteroid destroys the incoming asteroid
            else:
                # Both asteroids destroy each other
                a = 0  # Marking a as 0 to avoid adding it back to the stack
                stack.pop()  # destroind top of the stack
        if a != 0:
            # Add Surviving Asteroids to the Stack,
            # if it was not destroyed in a collision (a != 0)
            stack.append(a)
    return stack


def asteroid_collision_2(asteroids: List[int]) -> List[int]:
    stack = []
    for a in asteroids:
        while stack and a < 0 and stack[-1] > 0:
            # colision
            diff = a + stack[-1]
            match diff:
                case 0:
                    # Both asteroids destroy each other
                    stack.pop()
                    a = 0  # Marking a as 0 to avoid adding it back to the stack
                    break
                case _ if diff < 0:
                    # Incoming asteroid destroys the top of the stack asteroid
                    stack.pop()
                    continue  # Continue checking in case there are more collisions
                case _ if diff > 0:
                    # Top of the stack asteroid destroys the incoming asteroid
                    a = 0  # Marking a as 0 to avoid adding it back to the stack
                    break
        if a != 0:
            # Add Surviving Asteroids to the Stack,
            # if it was not destroyed in a collision (a != 0)
            stack.append(a)
    return stack
