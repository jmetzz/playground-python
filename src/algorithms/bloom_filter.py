from typing import Any, List, Optional


class BloomFilter:
    """
    A Bloom filter is a space-efficient probabilistic data structure,
    conceived by Burton Howard Bloom in 1970, that is used to test whether
    an element is a member of a set. False positive matches are possible,
    but false negatives are not
    """

    def __init__(self, size: int = 100, hash_count: int = 3):
        self._bit_array = [0] * size
        self.hash_count = hash_count

    def check(self, item) -> bool:
        hashes = self._get_hashes(item)
        return all([self._bit_array[h] for h in hashes])

    def add(self, item) -> None:
        hashes = self._get_hashes(item)
        for h in hashes:
            self._bit_array[h] = 1

    def _hash(self, item, seed: int) -> int:
        # This is a simple hash function for demonstration;
        # It uses the builtin hash from python standard library.
        # In practice, use a better hash function.
        hash_val = hash(item + str(seed))
        return hash_val % len(self._bit_array)

    def _get_hashes(self, item) -> List[int]:
        return [self._hash(item, h) for h in range(self.hash_count)]


class SimpleCache:
    def __init__(self):
        self.cache = {}

    def get_item(self, key):
        return self.cache.get(key, None)

    def add(self, key, value):
        self.cache[key] = value


def get_item_from_cache(key: str, cache_filter: BloomFilter, cache: SimpleCache) -> Optional[Any]:
    if not cache_filter.check(key):
        # Item definitely not in cache or dataset
        return None
    # Step 2: Check Cache
    return cache.get_item(key)


def database_lookup(key):
    match key:
        case "apple":
            return "horse"
        case "banana":
            return "monkey"
        case "grape":
            return "bird"
        case "orange":
            return "orangutan"
        case "watermelon":
            return "elephant"
        case _:
            return None


if __name__ == "__main__":
    # ---- Testing bloom filter ---
    # Create a Bloom Filter with a size of 1000 bits and 3 hash functions
    bloom_filter = BloomFilter(size=1000, hash_count=3)

    # List of items to add to the Bloom Filter
    items_to_add = ["apple", "banana", "grape", "orange", "watermelon"]

    # Add items to the Bloom Filter
    for i in items_to_add:
        bloom_filter.add(i)
        print(f"Added {i} to the Bloom Filter.")

    # List of items to check in the Bloom Filter
    items_to_check = ["apple", "banana", "cherry", "date"]

    # Check if these items are in the Bloom Filter
    for i in items_to_check:
        result = "probably in" if bloom_filter.check(i) else "definitely not in"
        print(f"The item {i} is {result} the Bloom Filter.")

    # Using a bloom filter "in front of" a cache
    simple_cache = SimpleCache()
    target_key = "apple"
    obj = get_item_from_cache(target_key, bloom_filter, simple_cache)
    if obj is None:
        # Item definitely not in cache or dataset.
        # Perform a database lookup
        database_result = database_lookup(target_key)
        if database_result:
            # Step 4: Update Cache and Bloom Filter
            simple_cache.add(target_key, database_result)
            bloom_filter.add(target_key)
        print(database_result)

    obj = get_item_from_cache("mango", bloom_filter, simple_cache)
    print(obj)
