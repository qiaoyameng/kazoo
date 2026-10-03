"""Tests for persistent watch event fan-out in the connection handler.

No ZooKeeper server is required: watch events are fed in as the fixed
bytes of a WatcherEvent packet and dispatched through the real
``ConnectionHandler._read_watch_event`` logic with a minimal client
double.
"""

from __future__ import annotations

from collections import defaultdict
import threading
import unittest

from kazoo.protocol.connection import ConnectionHandler
from kazoo.protocol.serialization import int_int_struct, write_string
from kazoo.protocol.states import (
    AddWatchMode,
    EventType,
    KeeperState,
)
from kazoo.handlers.threading import SequentialThreadingHandler


def watch_event_bytes(
    event_type: int, path: str, state: int = 3
) -> bytes:
    return int_int_struct.pack(event_type, state) + write_string(path)


class FakeClient:
    """The subset of KazooClient used by _read_watch_event."""

    def __init__(self, chroot: str = "") -> None:
        from kazoo.protocol.persistent import PersistentWatchManager

        self.handler = SequentialThreadingHandler()
        self.handler.start()
        self._data_watchers: defaultdict[
            str, set
        ] = defaultdict(set)
        self._child_watchers: defaultdict[
            str, set
        ] = defaultdict(set)
        self._persistent_watchers = PersistentWatchManager()
        self._state = KeeperState.CONNECTED
        self._stopped = threading.Event()
        self.chroot = chroot

    def unchroot(self, path: str) -> str:
        if not self.chroot:
            return path
        if self.chroot == path:
            return "/"
        return path[len(self.chroot):] if path.startswith(self.chroot) else path


class FanOutTestBase(unittest.TestCase):
    chroot = ""

    def setUp(self) -> None:
        self.client = FakeClient(chroot=self.chroot)
        self.handler = ConnectionHandler.__new__(ConnectionHandler)
        self.handler.client = self.client  # type: ignore[assignment]
        self.handler.logger = __import__("logging").getLogger(
            "test-fanout"
        )

    def tearDown(self) -> None:
        self.client.handler.stop()

    def feed(self, event_type: int, path: str) -> None:
        payload = watch_event_bytes(event_type, path)
        self.handler._read_watch_event(payload, 0)

    def drain(self) -> None:
        # Watch callbacks run on the handler's watch queue. A short sleep
        # is sufficient because the queue is drained sequentially in a
        # dedicated thread.
        threading.Event().wait(0.1)


class TestPersistentDelivery(FanOutTestBase):
    def test_persistent_watch_fires_repeatedly(self) -> None:
        calls: list[EventType] = []
        fired = threading.Event()

        def cb(event: object) -> None:
            ev_type = getattr(event, "type")
            calls.append(ev_type)
            fired.set()

        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, cb
        )

        for _ in range(3):
            fired.clear()
            self.feed(3, "/app")
            assert fired.wait(2)

        assert calls == [EventType.CHANGED] * 3

    def test_persistent_watch_not_consumed_while_one_shot_is(self) -> None:
        cb_calls: list[int] = []
        once_calls: list[int] = []

        def persistent_cb(event: object) -> None:
            cb_calls.append(1)

        def once_cb(event: object) -> None:
            once_calls.append(1)

        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, persistent_cb
        )
        self.client._data_watchers["/app"].add(once_cb)

        self.feed(3, "/app")
        self.feed(3, "/app")
        self.drain()

        assert len(cb_calls) == 2
        assert len(once_calls) == 1

    def test_same_callback_used_for_one_shot_and_persistent_called_once(
        self,
    ) -> None:
        calls: list[int] = []

        def cb(event: object) -> None:
            calls.append(1)

        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, cb
        )
        self.client._data_watchers["/app"].add(cb)

        self.feed(3, "/app")
        self.drain()
        assert len(calls) == 1

    def test_recursive_watch_fans_out_to_subtree_only(self) -> None:
        events: list[str] = []

        def cb(event: object) -> None:
            events.append(getattr(event, "path"))

        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT_RECURSIVE, cb
        )

        self.feed(3, "/app")
        self.feed(3, "/app/sub")
        self.feed(1, "/app/sub/deep")
        self.feed(4, "/app/sub")
        self.feed(3, "/apple")
        self.drain()

        assert events == ["/app", "/app/sub", "/app/sub/deep", "/app/sub"]

    def test_overlapping_registrations_invoke_callback_once(self) -> None:
        calls: list[int] = []

        def cb(event: object) -> None:
            calls.append(1)

        self.client._persistent_watchers.add(
            "/", AddWatchMode.PERSISTENT_RECURSIVE, cb
        )
        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT_RECURSIVE, cb
        )
        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, cb
        )

        self.feed(3, "/app")
        self.drain()
        assert len(calls) == 1

    def test_deleted_event_dedup_across_data_and_child_one_shots(
        self,
    ) -> None:
        calls: list[int] = []

        def cb(event: object) -> None:
            calls.append(1)

        self.client._data_watchers["/app"].add(cb)
        self.client._child_watchers["/app"].add(cb)

        self.feed(2, "/app")
        self.drain()
        assert len(calls) == 1

    def test_one_shot_callbacks_run_before_persistent_ones(self) -> None:
        order: list[str] = []

        def once(event: object) -> None:
            order.append("once")

        def persistent(event: object) -> None:
            order.append("persistent")

        self.client._data_watchers["/app"].add(once)
        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, persistent
        )

        self.feed(3, "/app")
        self.drain()
        assert order == ["once", "persistent"]


class TestPersistentWatchRemoved(FanOutTestBase):
    def test_removed_event_notifies_and_drops_registration(self) -> None:
        seen: list[EventType] = []
        done = threading.Event()

        def cb(event: object) -> None:
            seen.append(getattr(event, "type"))
            done.set()

        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, cb
        )
        self.feed(7, "/app")
        assert done.wait(2)

        self.feed(3, "/app")
        self.drain()
        assert seen == [EventType.PERSISTENT_WATCH_REMOVED]

    def test_removed_event_does_not_fan_out_to_recursive_ancestor(
        self,
    ) -> None:
        calls: list[int] = []

        def ancestor(event: object) -> None:
            calls.append(1)

        self.client._persistent_watchers.add(
            "/", AddWatchMode.PERSISTENT_RECURSIVE, ancestor
        )
        self.feed(7, "/app")
        self.drain()
        assert calls == []

    def test_removed_event_does_not_touch_one_shot_watches(self) -> None:
        one_shot_calls: list[int] = []
        persistent_calls: list[int] = []

        def once(event: object) -> None:
            one_shot_calls.append(1)

        def persistent(event: object) -> None:
            persistent_calls.append(1)

        self.client._data_watchers["/app"].add(once)
        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, persistent
        )

        self.feed(7, "/app")
        self.drain()
        assert one_shot_calls == []
        assert persistent_calls == [1]
        # The one-shot watch remains registered.
        assert once in self.client._data_watchers["/app"]


class TestStoppedClient(FanOutTestBase):
    def test_no_callbacks_after_stop(self) -> None:
        calls: list[int] = []

        def cb(event: object) -> None:
            calls.append(1)

        self.client._persistent_watchers.add(
            "/app", AddWatchMode.PERSISTENT, cb
        )
        self.client._stopped.set()
        self.feed(3, "/app")
        self.drain()
        assert calls == []


class TestOneShotUnchanged(FanOutTestBase):
    def test_data_watch_consumed_on_data_event(self) -> None:
        calls: list[int] = []

        def cb(event: object) -> None:
            calls.append(1)

        self.client._data_watchers["/app"].add(cb)
        self.feed(3, "/app")
        self.feed(3, "/app")
        self.drain()
        assert len(calls) == 1
        assert self.client._data_watchers.get("/app") is None

    def test_child_watch_consumed_on_child_event(self) -> None:
        calls: list[int] = []

        def cb(event: object) -> None:
            calls.append(1)

        self.client._child_watchers["/app"].add(cb)
        self.feed(4, "/app")
        self.feed(4, "/app")
        self.drain()
        assert len(calls) == 1

    def test_delete_clears_data_and_child_watches_once(self) -> None:
        seen: list[EventType] = []
        done = threading.Event()

        def cb(event: object) -> None:
            seen.append(getattr(event, "type"))
            done.set()

        self.client._data_watchers["/app"].add(cb)
        self.client._child_watchers["/app"].add(cb)

        self.feed(2, "/app")
        assert done.wait(2)
        assert seen == [EventType.DELETED]
        assert self.client._data_watchers.get("/app") is None
        assert self.client._child_watchers.get("/app") is None


class TestChrootFanOut(FanOutTestBase):
    chroot = "/chroot"

    def test_event_path_is_unrooted_for_callback(self) -> None:
        seen: list[str] = []
        done = threading.Event()

        def cb(event: object) -> None:
            seen.append(getattr(event, "path"))
            done.set()

        self.client._persistent_watchers.add(
            "/chroot/app", AddWatchMode.PERSISTENT_RECURSIVE, cb
        )
        self.feed(3, "/chroot/app/sub")
        assert done.wait(2)
        assert seen == ["/app/sub"]
