"""A disk cache in front of the hosted bio-knowledge MCP servers.

The biocurator is the only agent that leaves the cluster: it queries two HOSTED servers
(``runner.BIO_MCP_URLS`` -- BioContextAI and OmniPath) over the network. Those are third-party,
fair-use endpoints, so sweep concurrency is throttled to protect them rather than because the
cluster is short of cores. That throttle is the binding constraint on how many seeds a sweep can
run, and seed count is the binding constraint on whether a sweep can measure anything: bootstrapped
from the pooled luna/blasi results, 5 runs per arm gives **4%** power to detect a 20-BIC improvement
on pass/fail, and 20 runs per arm gives 80% on a rank test.

Caching removes the constraint almost entirely, because a sweep asks the SAME questions over and
over. The entity set is a property of the scenario, not of the seed -- every fujita run looks up
EGF, EGFR, AKT and ribosomal protein S6 -- so after one warm run the remaining seeds are served from
disk and never touch the network at all.

It also removes a source of experimental noise that has nothing to do with the thing under test: an
uncached sweep's biocurator sees whatever those servers return that day, and a transport failure
silently degrades that run to "dumb" (see ``bio_sessions``). A cached sweep replays identical
answers to every arm, which is what a controlled comparison needs.

DESIGN NOTES

* Keyed on ``(url, tool_name, arguments)`` with arguments canonicalised, so argument order cannot
  produce a spurious miss.
* Values are pickled ``CallToolResult`` objects rather than an extracted string. The consumers are
  langchain-mcp-adapters and the deepagents panel, which read fields beyond the text content; a
  hand-rolled stub would work until something read ``isError`` or ``structuredContent``. Any
  unpickling failure is treated as a MISS, so a library upgrade degrades to a network call rather
  than an exception.
* Writes are atomic (temp file + ``os.replace``): a sweep has many array tasks sharing one cache
  directory, and a torn read must be impossible.
* ERRORS ARE NEVER CACHED. A failed lookup is exactly what should be retried next time, and
  caching one would pin a transient outage into every later run of the sweep.
* The cache directory must be on SHARED storage to be worth anything -- the point is that task 2
  reads what task 1 wrote. It defaults next to the other sweep state and is overridable with
  ``PETAB_BIO_CACHE_DIR``; set that to an empty string to disable caching entirely.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

# Bumped when the stored representation changes in a way that makes old entries unreadable or
# wrong. Entries live in a subdirectory named for it, so a bump orphans the old ones rather than
# mixing them in — and the old directory can simply be deleted.
_CACHE_SCHEMA = "v1"

# Keyword arguments that change how a result is DELIVERED, not what it is, so a cached answer is
# still correct in their presence. Everything else bypasses the cache. `progress_callback` is the
# load-bearing entry: langchain-mcp-adapters passes it on every call.
_INERT_KWARGS = frozenset({"progress_callback", "read_timeout_seconds"})


def cache_dir() -> "Path | None":
    """Where entries live, or ``None`` when caching is switched off.

    ``PETAB_BIO_CACHE_DIR`` overrides the default; setting it to an empty string disables the cache,
    which is how a run verifies it still works against the live servers.
    """
    raw = os.environ.get("PETAB_BIO_CACHE_DIR")
    if raw is not None and not raw.strip():
        return None
    if raw:
        base = Path(raw).expanduser()
    else:
        # Via `paths.repo_root()`, not a private `parents[2]` walk: that module exists to be the one
        # rule for this and honours `PETAB_MCP_REPO_ROOT`, which array_task.sh sets on every task.
        # A hand-rolled walk lands in `<env>/lib` under a wheel install, giving a per-node cache
        # directory that is silently unshared -- which defeats the entire point, since the value
        # here is that task 2 reads what task 1 wrote.
        from .paths import repo_root

        base = repo_root() / "slurm" / "bio_cache"
    return base / _CACHE_SCHEMA


def _key(url: str, name: str, arguments: Any) -> str:
    """Stable digest of one call. Arguments are canonicalised (sorted keys, compact separators) so
    two logically identical calls cannot land on different keys."""
    try:
        args = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:  # noqa: BLE001 - an unserialisable argument still needs SOME stable key
        args = repr(arguments)
    return hashlib.sha256(f"{url}\x00{name}\x00{args}".encode()).hexdigest()


class BioCache:
    """Read-through cache for one bio MCP endpoint. Counters are for reporting the hit rate."""

    __slots__ = ("url", "dir", "hits", "misses", "writes", "errors", "bypasses")

    def __init__(self, url: str, directory: "Path | None") -> None:
        self.url = url
        self.dir = directory
        self.hits = self.misses = self.writes = self.errors = self.bypasses = 0

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _path(self, key: str) -> Path:
        # One level of fan-out: a sweep can produce thousands of entries, and some filesystems
        # (including several networked ones) degrade badly on a single huge directory.
        assert self.dir is not None
        return self.dir / key[:2] / f"{key}.pkl"

    def get(self, key: str) -> "tuple[bool, Any]":
        """``(hit, value)``. Any failure to read or unpickle is reported as a miss, never raised."""
        if not self.enabled:
            return False, None
        try:
            with open(self._path(key), "rb") as fh:
                value = pickle.load(fh)["result"]
        except FileNotFoundError:
            self.misses += 1
            return False, None
        except Exception:  # noqa: BLE001 - corrupt/incompatible entry: fall through to the network
            self.errors += 1
            self.misses += 1
            return False, None
        self.hits += 1
        return True, value

    def put(self, key: str, result: Any, *, name: str, arguments: Any) -> None:
        """Store atomically. Best-effort: a cache that cannot be written must not break the run."""
        if not self.enabled:
            return
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "result": result,
                "url": self.url,
                "tool": name,
                "arguments": arguments,
                "stored_at": time.time(),
            }
            # Temp file in the SAME directory so `os.replace` is a rename within one filesystem,
            # which is the part that makes it atomic against the other array tasks.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:  # closes fd on success AND on error
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except BaseException:
                # Clean up the temp file, but never let the cleanup mask the original failure.
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            self.writes += 1
        except Exception:  # noqa: BLE001 - unwritable cache dir, full disk, races: all non-fatal
            self.errors += 1

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "url": self.url,
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "bypasses": self.bypasses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }


def wrap_session(
    session: Any,
    url: str,
    *,
    directory: "Path | None" = ...,
    log: Callable[[str], Any] | None = None,
) -> BioCache:
    """Shadow ``session.call_tool`` with a read-through cache. Returns the cache for its stats.

    Shadowing the method rather than wrapping the object keeps every existing consumer working
    untouched — the same approach `runner.clock` uses to meter tool calls, and for the same reason:
    every call routes through this one place, so nothing can bypass it.
    """
    cache = BioCache(url, cache_dir() if directory is ... else directory)
    if not cache.enabled:
        return cache
    # Nothing to shadow. A real MCP session always has this, but `open_bio_sessions` is deliberately
    # injectable (its supervision and teardown logic is unit-tested with stand-ins that implement
    # only what those tests need), and caching must never be the reason one of those breaks.
    original = getattr(session, "call_tool", None)
    if original is None:
        cache.dir = None
        return cache

    async def cached(name, arguments=None, *args, **kwargs):
        # Extra state generally makes a call uncacheable — a cached answer that ignored it would be
        # subtly wrong, worse than not caching. But TRANSPORT arguments are not extra state: they
        # affect how the answer is delivered, not what it is, so bypassing on them disables the
        # cache entirely. That is not hypothetical: langchain-mcp-adapters (the ONLY consumer of
        # these sessions, via `load_mcp_tools`) passes `progress_callback=` on every single call
        # (tools.py:489), so a blanket `if kwargs: bypass` produced a 0% hit rate in production
        # while every unit test — which called the stand-in in the plain shape — passed.
        if args or (set(kwargs) - _INERT_KWARGS):
            # Counted: without this, a future library release adding one kwarg would bypass every
            # call and log "0 hits / 0 misses (rate None)" — byte-identical to a run whose
            # biocurator never queried anything. That is how the first version hid a 0% hit rate.
            cache.bypasses += 1
            return await original(name, arguments, *args, **kwargs)
        # A non-None read timeout is a real constraint on the call and is NOT inert.
        if kwargs.get("read_timeout_seconds") is not None:
            cache.bypasses += 1
            return await original(name, arguments, *args, **kwargs)
        key = _key(url, str(name), arguments)
        hit, value = cache.get(key)
        if hit:
            return value
        result = await original(name, arguments)
        # Never cache an error: a failed lookup is what should be retried, and pinning a transient
        # outage into the cache would degrade every later run of the sweep.
        if not getattr(result, "isError", False):
            cache.put(key, result, name=str(name), arguments=arguments)
        return result

    try:
        session.call_tool = cached  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001 - a frozen session simply goes uncached
        # Disable it, or the run logs an ACTIVE cache reporting "0 hits / 0 misses" — the exact
        # false reassurance the hit-rate logging exists to prevent. The missing-`call_tool` branch
        # above already does this; this one did not.
        cache.dir = None
        if log:
            log(f"[bio_cache] could not wrap {url}; proceeding uncached")
    return cache
