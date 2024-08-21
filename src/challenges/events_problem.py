# ruff: disable-file

"""We're building a small service for drawing bar charts of user activity over time.
There are two operations on that service:
add_activity(user_id: str, activity_timestamp: int)
barchart_heights(user_id: str,
                from_timestamp: int,
                to_timestamp: int,
                frequency: Literal["minute"|"hour"|"day"]
                ) -> List[int]

add_activity records user activity (we don't really care what it is)
that we can query later.

barchart_heights returns heights of the bars that we have to draw,
for the user activity between from_timestamp and to_timestamp.
Observe that bars can represent minutes, hours, or days.

Example:
-------
add_activity("user_1", 10)
add_activity("user_1", 12)
add_activity("user_1", 17)
add_activity("user_1", 60)

 add_activity("user_2", 61)
 add_activity("user_1", 70)
 barchart_heights("user_1", 5, 130, frequency="minute")
 # -> [4, 1, 0]
 # There are 4 activities between seconds 5 and 65 [10, 12, 17, 60]
 # There's one activity between seconds 65 and 125 [70]
 # There is no activity between seconds 125 and 130.

"""

import math
from collections import Counter
from typing import List, Literal

DATABASE = {}  # {user_id: [ts1, ts2]}


def add_activity(user_id: str, activity_timestamp: int) -> None:
    if (user_id, activity_timestamp) in DATABASE:
        raise ValueError("Colision: User and activity alredy exist.Pais " "({user_id}, {activiti_timestamp})")
    DATABASE[(user_id, activity_timestamp)] = "event"


def fetch_activities(user_id):
    DATABASE[user_id]


def calc_offset(timestamp):
    pass


def barchart_heights(
    user_id: str, from_timestamp: int, to_timestamp: int, frequency: Literal["minute" | "hour" | "day"]
) -> List[int]:
    #     raise ValueError("Invalid frequency unit")

    time_delta = to_timestamp - from_timestamp
    conversion_map = {"minute": 60, "hour": 60 * 60, "day": 60 * 60 * 24}

    events = fetch_activities(user_id, from_timestamp, to_timestamp)  # the list of time_stamp events
    # [10, 12, 17, 60, 70]
    # counter -> {10: 1, 12:1, 17: 1, 60:1, 62:1, 70:1}
    # time delta: 125
    # time_buckets len: 2 --> [4 , 1, 0]
    # O(nlogn + k) + O(k)

    counter = Counter(events)  # how many event in each time_step -->t1: 4, t2: 2
    time_buckets = [0] * len(math.ceil(time_delta / conversion_map[frequency]))
    # would return the start of previous minute from_timestamp
    off_set = calc_offset(from_timestamp)
    for k, v in counter.items():
        time_buckets[(k - off_set) // conversion_map[frequency]] += v

    return time_buckets


if __name__ == "__main__":
    pass
