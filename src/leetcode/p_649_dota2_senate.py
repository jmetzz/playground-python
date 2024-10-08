"""https://leetcode.com/problems/dota2-senate/description

649. Dota2 Senate
Medium

In the world of Dota2, there are two parties: the Radiant and the Dire.

The Dota2 senate consists of senators coming from two parties.
Now the Senate wants to decide on a change in the Dota2 game.
The voting for this change is a round-based procedure.
In each round, each senator can exercise one of the two rights:

Ban one senator's right: A senator can make another senator lose
all his rights in this and all the following rounds.
Announce the victory: If this senator found the senators who still
have rights to vote are all from the same party, he can announce
the victory and decide on the change in the game.
Given a string senate representing each senator's party belonging.
The character 'R' and 'D' represent the Radiant party and the Dire party.
Then if there are n senators, the size of the given string will be n.

The round-based procedure starts from the first senator to the
last senator in the given order. This procedure will last until
the end of voting. All the senators who have lost their rights
will be skipped during the procedure.

Suppose every senator is smart enough and will play the best
strategy for his own party. Predict which party will finally
announce the victory and change the Dota2 game.
The output should be "Radiant" or "Dire".

Example 1:
Input: senate = "RD"
Output: "Radiant"
Explanation:
The first senator comes from Radiant and he can just ban the next senator's right in round 1.
And the second senator can't exercise any rights anymore since his right has been banned.
And in round 2, the first senator can just announce the victory since he is the only guy in the senate who can vote.

Example 2:
Input: senate = "RDD"
Output: "Dire"
Explanation:
The first senator comes from Radiant and he can just ban the next senator's right in round 1.
And the second senator can't exercise any rights anymore since his right has been banned.
And the third senator comes from Dire and he can ban the first senator's right in round 1.
And in round 2, the third senator can just announce the victory since he is the only guy in the senate who can vote.


Constraints:

n == senate.length
1 <= n <= 104
senate[i] is either 'R' or 'D'.
"""

from collections import deque


def predict_party_victory(senate: str) -> str:
    q = deque(senate)
    answers = {"R": "Radiant", "D": "Dire"}
    while q:
        party = q.popleft()
        if party == "*":
            # skip since this senator was already banned
            continue

        # try to ban opposition
        i = 0
        while i < len(q):
            if q[i] != party and q[i] != "*":
                q[i] = "*"  # ban the senator i
                q.append(party)  # acted, now go to the end of the queue
                break
            i += 1
        if i == len(q):
            # everybody as voted
            break
    return answers[party]


def predict_party_victory_double_deque(senate: str) -> str:
    """Rationale:

    - Use two queues to keep the active senator's positions.
        Each senator has a position in the input list, which
        defines the order to exercise their right.
        The ones to the left have an earlier turn than the ones to the right.
    - radiant_queue is the queue that holds positions (indices) of active senators in "Radiant"
    - dire_quere is the queue that holds all positions (indices) of active senators in "Dire".
    - Both queues will be ordered so that the senators with earlier voting power
      come first (to the left of the queue), and thus, have the opportunity to
      ban the next opposing senator.
    - Banned senators are not added back to the queue, and therefore, have lost the right to vote.
    - We keep doing this until one queue is empty which means there are no more senators on the team.
    """
    radiant_queue = deque()
    dire_queue = deque()
    for i, s in enumerate(senate):
        if s == "R":
            radiant_queue.append(i)
        else:
            dire_queue.append(i)
    n = len(senate)
    while radiant_queue and dire_queue:
        n += 1
        r = radiant_queue.popleft()
        d = dire_queue.popleft()
        if r < d:
            radiant_queue.append(n)
        else:
            dire_queue.append(n)
    return "Dire" if dire_queue else "Radiant"


inputs = ["RD", "RDD"]
for s in inputs:
    print(predict_party_victory(s))
    print(predict_party_victory_double_deque(s))
