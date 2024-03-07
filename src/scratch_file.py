from data_structures import TrieSymbolTableDict, TrieSymbolTableRecursive


def tryout_TrieSymbolTableRecursive():
    t = TrieSymbolTableRecursive()
    t.insert("apple", "apple")
    print(t.contains("apple"))  # expect: True
    print(t.contains("app"))  # expect: False
    print(t.keys_with_prefix("app"))  # expect: apple
    print(t.insert("app", "app"))
    print(t.contains("app"))  # expect: True

    print("-" * 20)
    t = TrieSymbolTableRecursive()
    print(t.get("apple"))  # expect: None
    t.insert("apple", "🍎")
    print(t.contains("apple"))  # expect: True
    print(t.contains("app"))  # expect: False
    print(t.get("apple"))  # expect: 🍎
    print(t.keys())  # expect: [apple]
    print(t.keys_with_prefix("app"))  # expect: [apple]
    t.insert("app", "📱")
    print(t.get("app"))  # expect: "📱"
    print(t.contains("app"))  # expect: True
    print(t.keys_with_prefix("app"))  # expect: [app, apple]
    print(t.size())  # expect: 2
    print(t.lazy_size())  # expect: 2
    print(t.keys())  # expect: [app, apple]

    print("-" * 20)
    print(t.is_valid_prefix("a"))  # expect: True
    print(t.is_valid_prefix("ap"))  # expect: True
    print(t.is_valid_prefix("app"))  # expect: True
    print(t.is_valid_prefix("appl"))  # expect: True
    print(t.is_valid_prefix("zeta"))  # expect: Faslse

    print("-" * 20 + "Deleting elements")
    t.delete("app")
    print(t.size())  # expect: 1
    print(t.lazy_size())  # expect: 1
    print(t.keys())  # expect: [apple]

    print("-" * 20)
    print(t.empty())  # expect: False
    t.delete("apple")
    print(t.keys())  # expect: []
    print(t.empty())  # expect: True
    print("-" * 20)

    # try to delete non-existing key
    t.delete("app")  # not exception should be triggers, but a log.warning()
    print(t.empty())  # expect: True
    print(t.keys())  # expect: [apple]


def tryout_TrieSymbolTableDict():
    t = TrieSymbolTableDict()

    print(t.get("apple"))  # expect: None
    t.insert("apple")
    print(t.contains("apple"))  # expect: True
    print(t.contains("app"))  # expect: False
    print(t.get("apple") is not None)  # expect: true
    print(t.get("app") is None)  # expect: true
    print(t.is_valid_prefix("app"))  # expect: true

    print(t.keys())  # expect: [apple]
    t.insert("app")
    print(t.keys())  # expect: [app, apple]
    print(t.get("app"))  # expect: the tree with prefix "app", ie, '<EndOfWord>' in it
    print(t.contains("app"))  # expect: True
    print(t.size())  # expect: 2

    print(f"Subtree 'ap': {t.get('ap')}")  # expect: None since "ap" is not a word in this Trie
    print(f"keys_with_prefix('ap'): {t.keys_with_prefix('ap')}")  # expect: [app, apple]

    t.delete("app")
    print(f"keys_with_prefix('ap'): {t.keys_with_prefix('ap')}")  # expect: [apple]
    print(t.keys())  # expect: [apple]
    print(t.size())  # expect: 1


if __name__ == "__main__":
    tryout_TrieSymbolTableDict()
