"""Lazy, bounded per-session dataset store (F-23).

A restarted server must not re-read every persisted session's CSV into memory. Each session gets a
``SessionDatasets`` view that remembers HOW to reload a dataset (path + recorded sha256) and loads the DataFrame
on first access into one shared LRU (``DatasetCache``) bounded across all sessions. A dataset set directly without
a registered path is pinned in the view (it cannot be reloaded, so it is never evicted).

A lazy load re-verifies the recorded sha256: a file changed since the startup integrity check is treated as
missing (``get`` -> None, ``in`` -> False) rather than analysed.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from typing import Any

log = logging.getLogger("pharmagent")


class DatasetCache:
    """Thread-safe LRU of loaded DataFrames keyed by (session id, dataset id)."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str]) -> Any | None:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def put(self, key: tuple[str, str], df: Any) -> None:
        with self._lock:
            self._items[key] = df
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def discard(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._items.pop(key, None)

    def keys(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._items)


class SessionDatasets(MutableMapping):
    def __init__(self, cache: DatasetCache, session_id: str,
                 reader: Callable[[str, str | None], Any]) -> None:
        self._cache = cache
        self._sid = session_id
        self._reader = reader                       # (path, session_id) -> DataFrame
        self._paths: dict[str, tuple[str, str | None]] = {}   # dsid -> (path, recorded sha256)
        self._pinned: dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, dsid: str, path: str, sha256: str | None = None, session_id: str | None = None) -> None:
        """Remember how to (re)load ``dsid``; makes it evictable. ``session_id`` is accepted for call symmetry."""
        with self._lock:
            self._paths[dsid] = (path, sha256)
            self._pinned.pop(dsid, None)

    def loaded_ids(self) -> list[str]:
        with self._lock:
            pinned = list(self._pinned)
        return pinned + [d for s, d in self._cache.keys() if s == self._sid]

    def _load(self, dsid: str) -> Any | None:
        cached = self._cache.get((self._sid, dsid))
        if cached is not None:
            return cached
        info = self._paths.get(dsid)
        if info is None:
            return None
        path, recorded = info
        try:
            if recorded:
                from app.core.provenance import file_sha256
                if file_sha256(path) != recorded:
                    log.warning("dataset %s of session %s changed on disk since startup; not loaded", dsid, self._sid)
                    return None
            df = self._reader(path, self._sid)
        except Exception:
            log.warning("could not reload dataset %s of session %s", dsid, self._sid, exc_info=True)
            return None
        self._cache.put((self._sid, dsid), df)
        return df

    def __getitem__(self, dsid: str) -> Any:
        with self._lock:
            if dsid in self._pinned:
                return self._pinned[dsid]
        df = self._load(dsid)
        if df is None:
            raise KeyError(dsid)
        return df

    def __setitem__(self, dsid: str, df: Any) -> None:
        with self._lock:
            registered = dsid in self._paths
            if not registered:
                self._pinned[dsid] = df
        if registered:
            self._cache.put((self._sid, dsid), df)

    def __delitem__(self, dsid: str) -> None:
        with self._lock:
            found = self._pinned.pop(dsid, None) is not None or self._paths.pop(dsid, None) is not None
        self._cache.discard((self._sid, dsid))
        if not found:
            raise KeyError(dsid)

    def __contains__(self, dsid: object) -> bool:
        try:
            self[dsid]  # type: ignore[index]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._pinned) + list(self._paths))

    def __len__(self) -> int:
        with self._lock:
            return len(set(self._pinned) | set(self._paths))
