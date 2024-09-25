"""The task is to implement a class to perform a search on a data source while filtering the data by a given tag.

Sample data:
{ "items": [
     {"name": "The Shawshank Redemption", "tags": ["90s", "drama"]},
     {"name": "The Godfather", "tags": ["70s", "drama", "crime"]},
     {"name": "The Dark Knight", "tags": ["action", "crime", "drama"]},
     {"name": "The Godfather: Part II", "tags": ["70s", "crime", "drama"]},
     ]
}


Implement two functions: search and first.

Search is a generator and return all matching results given the query tag.
Firts, returns the first element given the original order of the elements in the data.
If there is not item, StopIteration should be raised.

"""

import json
from pathlib import Path

SAMPLE_DATA_1 = {
    "items": [
        {"name": "The Shawshank Redemption", "tags": ["90s", "drama"]},
        {"name": "The Godfather", "tags": ["70s", "drama", "crime"]},
        {"name": "The Dark Knight", "tags": ["action", "crime", "drama"]},
        {"name": "The Godfather: Part II", "tags": ["70s", "crime", "drama"]},
    ]
}


SAMPLE_DATA_2 = {
    "items": [
        {"name": "The Shawshank Redemption"},
        {"name": "The Godfather"},
    ]
}


class SearchByTag:
    def __init__(self, data: list[dict], query_tag):
        self._items = data
        self.query = query_tag

        for item in self._items:
            if "tags" in item:
                item["tags"] = set(item["tags"])
            else:
                item["tags"] = set()

    def search(self):
        for item in self._items:
            if self.query in item["tags"]:
                yield item

    def first(self):
        element = next(self.search())
        if not element:
            raise StopIteration()
        return element


def test_with_sample_1(data, query):
    search_1 = SearchByTag(data, query)
    print(search_1.first())

    print("----\n")

    data = search_1.search()

    print(next(data))
    print(next(data))
    print("----\n")

    data_list = list(search_1.search())
    print(data_list)


def test_with_sample_2():
    search_2 = SearchByTag(SAMPLE_DATA_2["items"], "crime")
    try:
        print(search_2.first())
    except StopIteration:
        print("No matching element")


if __name__ == "__main__":
    test_with_sample_1(SAMPLE_DATA_1["items"], "crime")
    print("#" * 25 + "\n")

    test_with_sample_2()
    print("#" * 25 + "\n")

    movies_data = []
    file_obj = Path("resources/movies_example.json")
    try:
        with file_obj.open("r", encoding="utf-8") as data_file:
            movies_data = json.load(data_file)
    except FileNotFoundError:
        print(f"File not found: {file_obj.absolute()}")

    # print(movies_data)
    test_with_sample_1(movies_data["items"], "crime")
