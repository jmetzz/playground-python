import pytest
from algorithms.bloom_filter import BloomFilter, SimpleCache

pytestmark = pytest.mark.nondeterministic


def test_add_check_item():
    bf = BloomFilter(size=100, hash_count=3)
    item = "test_item"
    assert bf.check(item) is False, "Item should not be present before being added"
    bf.add(item)
    assert bf.check(item) is True, "Item should be present after being added"


def test_false_positive_rate():
    bf = BloomFilter(size=1000, hash_count=3)
    items_to_add = ["item1", "item2", "item3"]
    for item in items_to_add:
        bf.add(item)

    # Check for a non-added item; due to the nature of Bloom Filters,
    # this might still return True but should be unlikely with a low false positive rate
    assert bf.check("non_added_item") is False, "Expected false positive rate to be low"


def test_hash_uniformity():
    """This test is non-deterministic and might fail"""
    bf = BloomFilter(size=100, hash_count=3)
    item = "test_item"
    hashes = bf._get_hashes(item)
    unique_hashes = set(hashes)
    assert len(unique_hashes) == len(hashes), "Hash functions should generate unique values for a single item"


def test_cache_set_get():
    cache = SimpleCache()
    cache.add("key1", "value1")
    assert cache.get_item("key1") == "value1", "Cache should return the value that was set for 'key1'"


def test_cache_get_nonexistent():
    cache = SimpleCache()
    assert cache.get_item("nonexistent_key") is None, "Cache should return None for keys that have not been set"


def test_cache_overwrite_value():
    cache = SimpleCache()
    cache.add("key1", "value1")
    cache.add("key1", "value2")
    assert cache.get_item("key1") == "value2", "Cache should overwrite the existing value with the new value for 'key1'"


def test_cache_multiple_keys():
    cache = SimpleCache()
    cache.add("key1", "value1")
    cache.add("key2", "value2")
    assert cache.get_item("key1") == "value1", "Cache should return the correct value for 'key1'"
    assert cache.get_item("key2") == "value2", "Cache should return the correct value for 'key2'"
