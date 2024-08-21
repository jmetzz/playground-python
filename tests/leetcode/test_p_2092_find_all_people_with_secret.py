import pytest

from leetcode.p_2092_find_all_people_with_secret import FindAllPeopleWitSecret


@pytest.mark.parametrize(
    "n, meetings, first_person, expected",
    [
        (6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1, [0, 1, 2, 3, 5]),
        (4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3, [0, 1, 3]),
        (5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1, [0, 1, 2, 3, 4]),
        (6, [[0, 2, 1], [1, 3, 1], [4, 5, 1]], 1, [0, 1, 2, 3]),
    ],
)
def test_find_all_people_wit_secret_bfs(n, meetings, first_person, expected):
    assert FindAllPeopleWitSecret().solve_bfs(n, meetings, first_person) == expected


@pytest.mark.parametrize(
    "n, meetings, first_person, expected",
    [
        (6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1, [0, 1, 2, 3, 5]),
        (4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3, [0, 1, 3]),
        (5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1, [0, 1, 2, 3, 4]),
        (6, [[0, 2, 1], [1, 3, 1], [4, 5, 1]], 1, [0, 1, 2, 3]),
    ],
)
def test_find_all_people_wit_secret_dfs(n, meetings, first_person, expected):
    assert FindAllPeopleWitSecret().solve_dfs(n, meetings, first_person) == expected


@pytest.mark.parametrize(
    "n, meetings, first_person, expected",
    [
        (6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1, [0, 1, 2, 3, 5]),
        (4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3, [0, 1, 3]),
        (5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1, [0, 1, 2, 3, 4]),
        (6, [[0, 2, 1], [1, 3, 1], [4, 5, 1]], 1, [0, 1, 2, 3]),
    ],
)
def test_find_all_people_wit_secret_dfs_recursive(n, meetings, first_person, expected):
    assert FindAllPeopleWitSecret().solve_dfs_recursive(n, meetings, first_person) == expected


@pytest.mark.parametrize(
    "n, meetings, first_person, expected",
    [
        (6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1, [0, 1, 2, 3, 5]),
        (4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3, [0, 1, 3]),
        (5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1, [0, 1, 2, 3, 4]),
        (6, [[0, 2, 1], [1, 3, 1], [4, 5, 1]], 1, [0, 1, 2, 3]),
    ],
)
def test_find_all_people_wit_secret_earliest_inf_first(n, meetings, first_person, expected):
    assert FindAllPeopleWitSecret().solve_earliest_inf_first(n, meetings, first_person) == expected


@pytest.mark.parametrize(
    "n, meetings, first_person, expected",
    [
        (6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1, [0, 1, 2, 3, 5]),
        (4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3, [0, 1, 3]),
        (5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1, [0, 1, 2, 3, 4]),
        (6, [[0, 2, 1], [1, 3, 1], [4, 5, 1]], 1, [0, 1, 2, 3]),
    ],
)
def test_find_all_people_wit_secret_bfs_time_scale(n, meetings, first_person, expected):
    assert FindAllPeopleWitSecret().solve_bfs_time_scale(n, meetings, first_person) == expected


@pytest.mark.parametrize(
    "n, meetings, first_person, expected",
    [
        (6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1, [0, 1, 2, 3, 5]),
        (4, [[3, 1, 3], [1, 2, 2], [0, 3, 3]], 3, [0, 1, 3]),
        (5, [[3, 4, 2], [1, 2, 1], [2, 3, 1]], 1, [0, 1, 2, 3, 4]),
        (6, [[0, 2, 1], [1, 3, 1], [4, 5, 1]], 1, [0, 1, 2, 3]),
    ],
)
def test_find_all_people_wit_secret_union_find_reset(n, meetings, first_person, expected):
    assert FindAllPeopleWitSecret().solve_union_find_reset(n, meetings, first_person) == expected
