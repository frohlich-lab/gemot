"""The disk cache in front of the hosted bio-knowledge MCP servers.

The cache exists to remove the fair-use throttle that caps sweep concurrency, because seed count is
what decides whether a sweep can measure anything (5 runs/arm = 4% power on pass/fail). So the
properties under test are the ones that make it safe to rely on: identical calls hit, different
calls do not, errors are never pinned, a broken cache degrades to a network call, and concurrent
array tasks sharing one directory cannot read a torn entry.
"""

from __future__ import annotations

import asyncio
import pickle

import pytest

from petab_harness import bio_cache


class _Result:
    """Stand-in for an MCP CallToolResult — pickled whole, so it need only be picklable."""

    def __init__(self, text: str, is_error: bool = False) -> None:
        self.text = text
        self.isError = is_error

    def __eq__(self, other) -> bool:
        return isinstance(other, _Result) and (self.text, self.isError) == (other.text, other.isError)


class _Session:
    """Counts network calls so a hit can be distinguished from a silent re-fetch."""

    def __init__(self, result=None, error=False) -> None:
        self.calls: list[tuple] = []
        self._result = result or _Result("payload")
        self._error = error

    async def call_tool(self, name, arguments=None, *args, **kwargs):
        self.calls.append((name, arguments, args, kwargs))
        if self._error:
            return _Result("boom", is_error=True)
        return self._result


def _wrap(session, tmp_path, url="https://example/mcp"):
    return bio_cache.wrap_session(session, url, directory=tmp_path)


def test_the_second_identical_call_never_reaches_the_network(tmp_path):
    s = _Session()
    cache = _wrap(s, tmp_path)
    a = asyncio.run(s.call_tool("get_uniprot", {"symbol": "EGFR"}))
    b = asyncio.run(s.call_tool("get_uniprot", {"symbol": "EGFR"}))
    assert a == b == _Result("payload")
    assert len(s.calls) == 1, "the second call went to the network"
    assert (cache.hits, cache.misses, cache.writes) == (1, 1, 1)


def test_argument_order_does_not_cause_a_miss(tmp_path):
    """Keys are canonicalised. Without that, two logically identical lookups written in a different
    order would each pay a network round trip — which is most of what the cache is for."""
    s = _Session()
    _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"a": 1, "b": 2}))
    asyncio.run(s.call_tool("q", {"b": 2, "a": 1}))
    assert len(s.calls) == 1


def test_different_arguments_tools_and_urls_are_separate_entries(tmp_path):
    s = _Session()
    _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"symbol": "EGFR"}))
    asyncio.run(s.call_tool("q", {"symbol": "AKT1"}))     # different args
    asyncio.run(s.call_tool("other", {"symbol": "EGFR"}))  # different tool
    assert len(s.calls) == 3
    # ...and a different endpoint must not read another endpoint's answers.
    s2 = _Session(_Result("from-the-other-server"))
    _wrap(s2, tmp_path, url="https://other/mcp")
    got = asyncio.run(s2.call_tool("q", {"symbol": "EGFR"}))
    assert got == _Result("from-the-other-server") and len(s2.calls) == 1


def test_an_error_result_is_never_cached(tmp_path):
    """A failed lookup is exactly what should be retried. Caching one would pin a transient outage
    into every later run of the sweep — and the biocurator degrades silently on failure, so nobody
    would notice it had gone permanently dumb."""
    s = _Session(error=True)
    cache = _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"symbol": "EGFR"}))
    asyncio.run(s.call_tool("q", {"symbol": "EGFR"}))
    assert len(s.calls) == 2, "an error was served from cache"
    assert cache.writes == 0


def test_a_corrupt_entry_degrades_to_a_network_call(tmp_path):
    """A cache that raises is worse than no cache. A truncated or version-incompatible entry must
    read as a MISS."""
    s = _Session()
    cache = _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"symbol": "EGFR"}))
    for p in tmp_path.rglob("*.pkl"):
        p.write_bytes(b"not a pickle")
    got = asyncio.run(s.call_tool("q", {"symbol": "EGFR"}))
    assert got == _Result("payload")
    assert len(s.calls) == 2 and cache.errors >= 1


def test_an_unwritable_cache_directory_does_not_break_the_run(tmp_path):
    target = tmp_path / "nope"
    target.write_text("this is a file, not a directory")
    s = _Session()
    cache = bio_cache.wrap_session(s, "https://example/mcp", directory=target)
    assert asyncio.run(s.call_tool("q", {"x": 1})) == _Result("payload")
    assert cache.errors >= 1          # noticed...
    assert len(s.calls) == 1          # ...and the call still happened


def test_the_production_call_shape_is_cached(tmp_path):
    """THE test this file was missing. langchain-mcp-adapters is the only consumer of these sessions
    and passes `progress_callback=` on EVERY call (tools.py:489). The first version bypassed the
    cache on any kwarg, so it served 0% in production while every test here — which called the
    stand-in in the plain shape — passed. Measured before the fix: 3 identical langchain-shaped
    calls produced 3 network calls and 0 hits."""
    s = _Session()
    cache = _wrap(s, tmp_path)
    for _ in range(3):
        asyncio.run(s.call_tool("get_uniprot", {"symbol": "EGFR"},
                                progress_callback=lambda *a, **k: None))
    assert len(s.calls) == 1, "the production call shape still bypasses the cache"
    assert cache.hits == 2


def test_a_real_read_timeout_still_bypasses_the_cache(tmp_path):
    """Inert transport kwargs are ignored, but a genuine constraint is not: `read_timeout_seconds`
    set to a value is a real bound on the call, so it must not be served from cache."""
    s = _Session()
    _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"x": 1}))
    asyncio.run(s.call_tool("q", {"x": 1}, read_timeout_seconds=5))
    assert len(s.calls) == 2
    # ...while an explicit None is inert and must NOT defeat the cache.
    asyncio.run(s.call_tool("q", {"x": 1}, read_timeout_seconds=None))
    assert len(s.calls) == 2


def test_an_unknown_keyword_still_bypasses_the_cache(tmp_path):
    """The allow-list is deliberately small: an argument nobody has reasoned about could change the
    RESULT, and a cached answer that ignored it would be subtly wrong."""
    s = _Session()
    _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"x": 1}))
    asyncio.run(s.call_tool("q", {"x": 1}, some_future_option=True))
    assert len(s.calls) == 2


def test_writes_are_atomic_so_concurrent_tasks_cannot_read_a_torn_entry(tmp_path):
    """A sweep runs many array tasks against ONE shared cache directory. Every committed file must
    be a complete pickle — asserted by writing concurrently and reading every entry back."""
    async def hammer():
        # SAME key on every session, so all 12 writers collide on ONE file — the contention the
        # atomic rename exists for. Giving each a different key (as this test first did) exercises
        # nothing: removing `os.replace` would still have passed.
        sessions = [_Session(_Result("shared-payload")) for _ in range(12)]
        for s in sessions:
            _wrap(s, tmp_path)
        await asyncio.gather(*(s.call_tool("q", {"same": "key"}) for s in sessions))
        again = await asyncio.gather(*(s.call_tool("q", {"same": "key"}) for s in sessions))
        return sessions, again

    sessions, again = asyncio.run(hammer())
    assert all(r.text == "shared-payload" for r in again), "a torn or missing entry was read back"
    assert len(list(tmp_path.rglob("*.pkl"))) == 1, "one key must produce exactly one entry"
    for p in tmp_path.rglob("*.pkl"):
        with open(p, "rb") as fh:
            assert "result" in pickle.load(fh)          # every committed file is complete
    assert not list(tmp_path.rglob("*.tmp")), "a temp file was left behind"


def test_caching_can_be_switched_off(tmp_path, monkeypatch):
    """Set the env var empty to verify a run still works against the live servers — without that
    escape hatch a stale cache would be indistinguishable from a working one."""
    monkeypatch.setenv("PETAB_BIO_CACHE_DIR", "")
    assert bio_cache.cache_dir() is None
    s = _Session()
    cache = bio_cache.wrap_session(s, "https://example/mcp")
    assert cache.enabled is False
    asyncio.run(s.call_tool("q", {"x": 1}))
    asyncio.run(s.call_tool("q", {"x": 1}))
    assert len(s.calls) == 2


def test_the_cache_directory_is_versioned_by_schema(tmp_path, monkeypatch):
    """Entries live under a schema segment, so a representation change orphans old entries instead
    of mixing them in and reading one as if it were the new shape."""
    monkeypatch.setenv("PETAB_BIO_CACHE_DIR", str(tmp_path))
    assert bio_cache.cache_dir().name == bio_cache._CACHE_SCHEMA
    assert bio_cache.cache_dir().parent == tmp_path


def test_stats_report_the_hit_rate(tmp_path):
    s = _Session()
    cache = _wrap(s, tmp_path)
    asyncio.run(s.call_tool("q", {"x": 1}))
    asyncio.run(s.call_tool("q", {"x": 1}))
    st = cache.stats()
    assert st["hits"] == 1 and st["misses"] == 1 and st["hit_rate"] == 0.5
    assert st["enabled"] is True and st["url"] == "https://example/mcp"


def test_a_session_without_call_tool_is_left_alone(tmp_path):
    """`open_bio_sessions` is injectable and its supervision tests use stand-ins that implement only
    what they need. Caching must never be the reason one of those breaks — nor a real session that
    exposes a different surface."""
    class _Bare:
        pass

    bare = _Bare()
    cache = bio_cache.wrap_session(bare, "https://example/mcp", directory=tmp_path)
    assert cache.enabled is False
    assert not hasattr(bare, "call_tool")
