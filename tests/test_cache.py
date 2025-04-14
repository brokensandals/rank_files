from rank_files.cache import Cache


def test_cache(tmpdir):
    path = tmpdir / "testcache.sqlite3"
    with Cache(path) as cache:
        assert cache.fetch("foo") is None
        cache.put("foo", "bar")
        assert cache.fetch("foo") == "bar"
        assert cache.total_hits == 1
        assert cache.fetch("bar") is None
        cache.put("foo", "baz")
        assert cache.fetch("foo") == "baz"
        assert cache.total_hits == 2
        cache.put("other", "bar")
        assert cache.fetch("foo") == "baz"
        assert cache.total_hits == 3
        assert cache.fetch("other") == "bar"
        assert cache.total_hits == 4
