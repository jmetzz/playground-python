"""
https://leetcode.com/problems/open-the-lock/description/
752. Open the Lock
Medium

You have a lock in front of you with 4 circular wheels.
Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'.
The wheels can rotate freely and wrap around: for example we can turn '9'
to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of
the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays
any of these codes, the wheels of the lock will stop turning and you
will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock,
return the minimum total number of turns required to open the lock,
or -1 if it is impossible.


Example 1:
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".

Example 2:
Input: deadends = ["8888"], target = "0009"
Output: 1
Explanation: We can turn the last wheel in reverse to move from "0000" -> "0009".

Example 3:
Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1
Explanation: We cannot reach the target without getting stuck.


Constraints:

1 <= deadends.length <= 500
deadends[i].length == 4
target.length == 4
target will not be in the list deadends.
target and deadends[i] consist of digits only.

Hint 1
We can think of this problem as a shortest path problem on a graph:
    there are `10000` nodes (strings `'0000'` to `'9999'`), and there is an edge between
    two nodes if they differ in one digit,
    that digit differs by 1 (wrapping around, so `'0'` and `'9'` differ by 1),
    and if *both* nodes are not in `deadends`.
"""

from collections import deque
from typing import List


def open_lock_str(deadends: List[str], target: str) -> int:
    def append_new_whell_combinations(comb):
        for whell in range(4):
            next_comb = list(comb)
            next_comb[whell] = next_slot[next_comb[whell]]
            new_comb_str = "".join(next_comb)
            if new_comb_str not in visited:
                queue.append(new_comb_str)
                visited.add(new_comb_str)

            prev_comb = list(comb)
            prev_comb[whell] = prev_slot[prev_comb[whell]]
            new_comb_str = "".join(prev_comb)
            if new_comb_str not in visited:
                queue.append(new_comb_str)
                visited.add(new_comb_str)

    # mark deadend as visited combinations since we can't proceed further from them.
    visited = set(deadends)
    if "0000" in visited:
        return -1

    visited.add("0000")
    queue = deque(["0000"])

    next_slot = {str(d): str((d + 1) % 10) for d in range(10)}
    prev_slot = {str(d): str((d - 1) % 10) for d in range(10)}
    turns = 0
    while queue:
        for _ in range(len(queue)):
            curr_comb = queue.popleft()
            if curr_comb == target:
                return turns
            append_new_whell_combinations(curr_comb)
        turns += 1

    return -1


def open_lock_tuple(deadends: List[str], target: str) -> int:
    target = tuple(map(int, target))
    start_comb = (0, 0, 0, 0)

    queue = deque()
    queue.append(start_comb)

    visited = set([tuple(map(int, lock)) for lock in deadends])
    turns = 0
    while queue:
        for _ in range(len(queue)):
            curr_comb = queue.popleft()

            if curr_comb == target:
                return turns
            if curr_comb in visited:
                continue

            visited.add(curr_comb)

            for whell in range(4):
                # turn the whell to the next value
                new_comb = list(curr_comb)
                new_comb[whell] = (new_comb[whell] + 1) % 10
                new_comb = tuple(new_comb)
                if new_comb not in visited:
                    queue.append(new_comb)

                # turn the whell to the previous value
                new_comb = list(curr_comb)
                new_comb[whell] = (new_comb[whell] - 1) % 10
                new_comb = tuple(new_comb)
                if new_comb not in visited:
                    queue.append(new_comb)
        turns += 1

    return -1


if __name__ == "__main__":
    print(open_lock_str(["0201", "0101", "0102", "1212", "2002"], "0202"))

    print(open_lock_tuple(["0201", "0101", "0102", "1212", "2002"], "0202"))
    print(open_lock_tuple(["8888"], "0009"))
    print(open_lock_tuple(["8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888"], "8888"))
