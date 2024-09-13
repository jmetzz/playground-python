import pytest

from leetcode.p_146_lru_cache import LRUCache, LRUCacheOrderedDict


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_initialization(cache_class):
    """Test cache initialization with given capacity."""
    cache = cache_class(2)
    assert len(cache._lru_cache) == 0, "Cache should start empty"
    assert cache._capacity == 2, "Cache capacity should be set correctly"


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_put_and_get(cache_class):
    """Test basic put and get functionality."""
    cache = cache_class(2)

    # Insert two items and retrieve them
    cache.put(1, 1)
    cache.put(2, 2)

    assert cache.get(1) == 1, "Get should return correct value for key 1"
    assert cache.get(2) == 2, "Get should return correct value for key 2"

    # Retrieve non-existent key
    assert cache.get(3) == -1, "Get should return -1 for non-existent key"


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_lru_eviction(cache_class):
    """Test eviction of least recently used items when capacity is exceeded."""
    cache = cache_class(2)

    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1, "Get should return correct value for key 1"

    # Insert third item, which should evict key 2 (LRU)
    cache.put(3, 3)

    assert cache.get(2) == -1, "Key 2 should be evicted (LRU)"
    assert cache.get(1) == 1, "Key 1 should still be in the cache"
    assert cache.get(3) == 3, "Key 3 should be in the cache"


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_lru_eviction_complex(cache_class):
    """Test LRU eviction behavior with multiple accesses."""
    cache = cache_class(2)

    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1, "Get should return correct value for key 1"

    # Access key 2, making it the most recently used
    assert cache.get(2) == 2, "Get should return correct value for key 2"

    # Insert a new key, which should now evict key 1
    cache.put(3, 3)

    assert cache.get(1) == -1, "Key 1 should be evicted (LRU)"
    assert cache.get(2) == 2, "Key 2 should still be in the cache"
    assert cache.get(3) == 3, "Key 3 should be in the cache"


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_update_existing_key(cache_class):
    """Test that updating an existing key doesn't evict items unnecessarily."""
    cache = cache_class(2)

    cache.put(1, 1)
    cache.put(2, 2)

    # Update key 1 and ensure nothing is evicted
    cache.put(1, 10)

    assert cache.get(1) == 10, "Key 1 should have updated value"
    assert cache.get(2) == 2, "Key 2 should still be in the cache"
    assert cache.get(3) == -1, "Key 3 should not be in the cache"


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_over_capacity(cache_class):
    """Test that cache behaves correctly when over capacity."""
    cache = cache_class(1)

    cache.put(1, 1)
    cache.put(2, 2)  # This should evict key 1

    assert cache.get(1) == -1, "Key 1 should be evicted"
    assert cache.get(2) == 2, "Key 2 should be in the cache"


@pytest.mark.parametrize("cache_class", [LRUCache, LRUCacheOrderedDict])
def test_zero_capacity(cache_class):
    """Test behavior with zero capacity cache."""
    cache = cache_class(0)

    cache.put(1, 1)
    assert cache.get(1) == -1, "Cache with zero capacity should not store anything"
