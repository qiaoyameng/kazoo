"""Local bookkeeping for ZooKeeper persistent watches.

Persistent watches (ZooKeeper 3.6+) are server-side watchers that stay
registered after firing, unlike the one-shot watches implied by the
``watch`` flag of the data/children APIs. This module keeps track of
which callbacks the client has registered for each ``(path, mode)`` so
that they can be (re-)registered after a (re)connect and fan out to the
correct callbacks when an event arrives.
"""

from __future__ import annotations

from collections import defaultdict
import threading
from typing import (
    Callable,
    DefaultDict,
    Dict,
    NamedTuple,
    Set,
    TYPE_CHECKING,
)

from kazoo.protocol.states import (
    AddWatchMode,
    EventType,
    watch_path_matches,
)

if TYPE_CHECKING:
    from kazoo.protocol.states import WatchedEvent


class PersistentRegistration(NamedTuple):
    """A server-side persistent watch and the local callbacks bound to it.

    ``mode`` is the :class:`AddWatchMode` used when registering the
    watch. A single server watch per ``(path, mode)`` backs any number
    of local callbacks, which is why the set is re-registered as one
    unit after a (re)connect.
    """

    path: str
    mode: AddWatchMode
    callbacks: Set[Callable[..., object]]


class PersistentWatchManager:
    """Thread-safe registry of persistent and persistent-recursive
    watches keyed by the server-side ``(path, mode)`` tuple.

    All methods that mutate the registries take the manager lock. The
    lock is re-entrant and intended to be shared with callers (such as
    the connection re-registration code) that need to atomically inspect
    and queue registration or removal requests. Holding the lock across
    request enqueuing guarantees that a removal issued while the client
    is re-establishing its watches cannot be reordered ahead of the
    AddWatch requests, i.e. a removed watch is never resurrected.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # mode -> path -> set of callbacks. Paths are server-side paths
        # (chroot included) as used on the wire.
        self._registrations: DefaultDict[
            AddWatchMode, DefaultDict[str, Set[Callable[..., object]]]
        ] = defaultdict(lambda: defaultdict(set))

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def add(
        self,
        path: str,
        mode: AddWatchMode,
        callback: Callable[..., object],
    ) -> bool:
        """Bind a callback to a persistent watch.

        Returns ``True`` when a new server-side watch must be sent and
        ``False`` when an identical ``(path, mode)`` watch is already
        registered locally, in which case the callback is merely added
        to the existing registration.
        """
        with self._lock:
            mode_reg = self._registrations[mode]
            existed = path in mode_reg
            mode_reg[path].add(callback)
            return not existed

    def remove(
        self,
        path: str,
        mode: AddWatchMode,
        callback: Callable[..., object],
    ) -> bool:
        """Detach a callback from a persistent watch.

        Returns ``True`` when the last callback was removed and the
        server-side watch for ``(path, mode)`` must be torn down.
        """
        with self._lock:
            mode_reg = self._registrations[mode]
            callbacks = mode_reg.get(path)
            if callbacks is None or callback not in callbacks:
                return False
            callbacks.discard(callback)
            if not callbacks:
                del mode_reg[path]
                if not mode_reg:
                    del self._registrations[mode]
                return True
            return False

    def discard_path(self, path: str) -> list[PersistentRegistration]:
        """Remove every registration stored for ``path``.

        Used to react to a server-initiated PersistentWatchRemoved
        event. Returns the removed registrations so the connection layer
        can notify their callbacks.
        """
        with self._lock:
            removed: list[PersistentRegistration] = []
            for mode, mode_reg in list(self._registrations.items()):
                callbacks = mode_reg.pop(path, None)
                if callbacks:
                    removed.append(
                        PersistentRegistration(path, mode, set(callbacks))
                    )
                    if not mode_reg:
                        del self._registrations[mode]
            return removed

    def clear(self) -> None:
        """Drop all registries (used when the client is stopped)."""
        with self._lock:
            self._registrations.clear()

    def registrations(self) -> list[PersistentRegistration]:
        """Snapshot of the current registrations for re-registration."""
        with self._lock:
            return [
                PersistentRegistration(path, mode, set(callbacks))
                for mode, mode_reg in self._registrations.items()
                for path, callbacks in mode_reg.items()
            ]

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._registrations)

    def fire(self, event: WatchedEvent) -> list[Callable[..., object]]:
        """Return the distinct callbacks that must be invoked for an
        incoming watch event.

        Each callback appears at most once in the result, even if it is
        registered at more than one path/mode that matches the event.
        Callbacks are returned in a deterministic order so that event
        delivery (and therefore test expectations) are stable.
        """
        with self._lock:
            matched: Dict[Callable[..., object], None] = {}

            persistent = self._registrations.get(AddWatchMode.PERSISTENT)
            recursive = self._registrations.get(
                AddWatchMode.PERSISTENT_RECURSIVE
            )

            # Type 7 is the explicit PersistentWatchRemoved signal. It
            # targets an exact registration path and must not fan out to
            # recursive ancestors or descendants.
            if event.type is EventType.PERSISTENT_WATCH_REMOVED:
                candidate_sets = [
                    callbacks
                    for callbacks in (
                        persistent.get(event.path) if persistent else None,
                        recursive.get(event.path) if recursive else None,
                    )
                    if callbacks
                ]
                for callbacks in candidate_sets:
                    for callback in sorted(
                        callbacks, key=self._callback_sort_key
                    ):
                        matched.setdefault(callback, None)
                return list(matched)

            # A non-recursive persistent watch only sees the exact path
            # and only data events (created/deleted/changed).
            if persistent is not None and event.type in _DATA_EVENTS:
                callbacks = persistent.get(event.path)
                if callbacks:
                    for callback in sorted(
                        callbacks, key=self._callback_sort_key
                    ):
                        matched.setdefault(callback, None)

            # Recursive watches cover the registration path and its whole
            # subtree and receive data and children events.
            if recursive is not None:
                for watch_path, callbacks in recursive.items():
                    if watch_path_matches(watch_path, event.path):
                        for callback in sorted(
                            callbacks, key=self._callback_sort_key
                        ):
                            matched.setdefault(callback, None)

            return list(matched)

    @staticmethod
    def _callback_sort_key(
        callback: Callable[..., object],
    ) -> int:
        return id(callback)


_DATA_EVENTS = frozenset(
    {
        EventType.CREATED,
        EventType.DELETED,
        EventType.CHANGED,
    }
)
