"""Tests for persistent and persistent-recursive watches.

These tests never talk to a ZooKeeper server. The protocol layer is
exercised with the fixed wire bytes produced by
:mod:`kazoo.protocol.serialization`, and watch events are fed in as
hand-built byte sequences, letting a deterministic event ordering drive
all the connection/session state transitions.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from kazoo.client import KazooClient
from kazoo.handlers.threading import SequentialThreadingHandler
from kazoo.protocol.connection import ConnectionHandler
from kazoo.protocol.serialization import (
    AddWatch,
    RemoveWatches,
    int_int_struct,
    write_string,
)
from kazoo.protocol.states import (
    EventType,
    KeeperState,
    WatcherType,
)
from kazoo.retry import KazooRetry

# ZK watch event state value for SyncConnected
_SYNC_CONNECTED = 3

CREATED = 1
DELETED = 2
CHANGED = 3
CHILD = 4


def watch_event_payload(
    event_type: int, path: str, state: int = _SYNC_CONNECTED
) -> bytes:
    """Build the exact on-wire body of a WatcherEvent (xid == -1)."""
    return int_int_struct.pack(event_type, state) + write_string(path)


def make_client() -> KazooClient:
    """A KazooClient with a live handler but no network connection."""
    client = KazooClient()
    # The callback queues need a running handler for dispatch_callback
    client.handler.start()
    return client


@pytest.fixture()
def client() -> Any:
    c = make_client()
    yield c
    c.handler.stop()


@pytest.fixture()
def connection(client: KazooClient) -> ConnectionHandler:
    conn = client._connection
    client._stopped.clear()  # pretend the connection loop is running
    return conn


class TestAddWatchLocal:
    def test_registers_offline(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        assert client.add_watch("/app", cb) is True
        assert client._persistent_watchers["/app"] == {
            cb: WatcherType.PERSISTENT_RECURSIVE
        }

    def test_default_mode_is_recursive(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb)
        assert (
            client._persistent_watchers["/app"][cb]
            == WatcherType.PERSISTENT_RECURSIVE
        )

    def test_explicit_persistent_mode(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        assert (
            client._persistent_watchers["/app"][cb]
            == WatcherType.PERSISTENT
        )

    def test_rejects_non_persistent_mode(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        with pytest.raises(ValueError):
            client.add_watch("/app", cb, WatcherType.DATA)
        with pytest.raises(ValueError):
            client.add_watch("/app", cb, WatcherType.CHILDREN)

    def test_rejects_conflicting_mode(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        with pytest.raises(ValueError):
            client.add_watch("/app", cb, WatcherType.PERSISTENT_RECURSIVE)

    def test_duplicate_registration_is_idempotent(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        bucket = client._persistent_watchers["/app"]
        assert len(bucket) == 1

    def test_validates_arguments(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        with pytest.raises(TypeError):
            client.add_watch(123, cb)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            client.add_watch("/app", "not-callable")  # type: ignore[arg-type]

    def test_connected_registration_queues_addwatch(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client._connection._write_sock = MagicMock()
        client._state = KeeperState.CONNECTED
        client._live.set()
        client.add_watch_async("/app", cb, WatcherType.PERSISTENT)
        request, _ = client._queue[-1]
        assert isinstance(request, AddWatch)
        assert request.path == "/app"
        assert request.watcher_type == -1

    def test_suspended_registration_is_local_only(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client._live.clear()
        result = client.add_watch_async("/app", cb)
        assert result.get_nowait() is True
        assert len(client._queue) == 0


class TestSegmentMatching:
    def test_exact_match_only(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        hits = [c for c, _ in client._persistent_watches_for_event("/app")]
        assert cb in hits
        assert client._persistent_watches_for_event("/application") == []
        assert client._persistent_watches_for_event("/app/child") == []
        # The canonical segment counter-example:
        assert client._persistent_watches_for_event("/apple") == []

    def test_recursive_match_by_segment(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT_RECURSIVE)
        matched = lambda p: cb in [  # noqa: E731
            c for c, _ in client._persistent_watches_for_event(p)
        ]
        assert matched("/app")
        assert matched("/app/child")
        assert matched("/app/child/grand")
        # Prefix but not a segment boundary must not match.
        assert not matched("/apple")
        assert not matched("/application/x")

    def test_root_recursive_matches_everything(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/", cb, WatcherType.PERSISTENT_RECURSIVE)
        for path in ("/", "/app", "/app/child", "/apple"):
            assert cb in [
                c for c, _ in client._persistent_watches_for_event(path)
            ]


class TestCallCountDeterminism:
    def test_same_callback_multiple_registrations_called_once(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/a", cb, WatcherType.PERSISTENT)
        client.add_watch("/", cb, WatcherType.PERSISTENT_RECURSIVE)
        matches = client._persistent_watches_for_event("/a")
        assert [c for c, _ in matches].count(cb) == 1

    def test_dedup_uses_first_path_in_deterministic_order(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/z", cb, WatcherType.PERSISTENT)
        client.add_watch("/", cb, WatcherType.PERSISTENT_RECURSIVE)
        matches = client._persistent_watches_for_event("/z")
        assert len(matches) == 1
        # "/" sorts before "/z" and wins the single invocation.
        assert matches[0][1] == "/"

    def test_distinct_callbacks_both_called(self, client: KazooClient) -> None:
        def cb1(event: Any) -> None: ...

        def cb2(event: Any) -> None: ...

        client.add_watch("/", cb1, WatcherType.PERSISTENT_RECURSIVE)
        client.add_watch("/a", cb2, WatcherType.PERSISTENT)
        callbacks = [c for c, _ in client._persistent_watches_for_event("/a")]
        assert cb1 in callbacks and cb2 in callbacks

    def test_order_stable_across_calls(self, client: KazooClient) -> None:
        def cb1(event: Any) -> None: ...

        def cb2(event: Any) -> None: ...

        client.add_watch("/b", cb1, WatcherType.PERSISTENT_RECURSIVE)
        client.add_watch("/a", cb2, WatcherType.PERSISTENT_RECURSIVE)
        first = client._persistent_watches_for_event("/x")
        second = client._persistent_watches_for_event("/x")
        assert first == second == []
        deep = client._persistent_watches_for_event("/a/1")
        assert [c for c, _ in deep] == [cb2]


class TestEventFanout:
    def test_persistent_watch_survives_multiple_events(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        received: list[tuple[str, EventType]] = []

        def cb(event: Any) -> None:
            received.append((event.path, event.type))

        client.add_watch("/app", cb, WatcherType.PERSISTENT)

        connection._read_watch_event(watch_event_payload(CHANGED, "/app"), 0)
        connection._read_watch_event(watch_event_payload(CHILD, "/app"), 0)
        import time

        time.sleep(0.3)
        assert received == [
            ("/app", EventType.CHANGED),
            ("/app", EventType.CHILD),
        ]
        # Registration remains in place.
        assert cb in client._persistent_watchers["/app"]

    def test_recursive_watch_receives_descendant_event(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        received: list[str] = []

        def cb(event: Any) -> None:
            received.append(event.path)

        client.add_watch("/app", cb, WatcherType.PERSISTENT_RECURSIVE)
        connection._read_watch_event(
            watch_event_payload(CREATED, "/app/child"), 0
        )
        import time

        time.sleep(0.3)
        assert received == ["/app/child"]

    def test_segment_prefix_does_not_match(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        received: list[str] = []

        def cb(event: Any) -> None:
            received.append(event.path)

        client.add_watch("/app", cb, WatcherType.PERSISTENT_RECURSIVE)
        connection._read_watch_event(watch_event_payload(CHANGED, "/apple"), 0)
        import time

        time.sleep(0.3)
        assert received == []

    def test_oneshot_watch_still_consumed(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        received: list[str] = []

        def cb(event: Any) -> None:
            received.append("oneshot")

        client._data_watchers["/app"].add(cb)
        connection._read_watch_event(watch_event_payload(CHANGED, "/app"), 0)
        connection._read_watch_event(watch_event_payload(CHANGED, "/app"), 0)
        import time

        time.sleep(0.3)
        assert received == ["oneshot"]
        assert "/app" not in client._data_watchers

    def test_oneshot_and_persistent_same_callback_dispatches_once(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        received: list[Any] = []

        def cb(event: Any) -> None:
            received.append(event)

        client._data_watchers["/app"].add(cb)
        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        connection._read_watch_event(watch_event_payload(CHANGED, "/app"), 0)
        import time

        time.sleep(0.3)
        assert len(received) == 1
        # Persistent registration survives even though one-shot fired.
        assert cb in client._persistent_watchers["/app"]

    def test_unknown_event_ignored(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        # type 99 is not a known watch event
        connection._read_watch_event(watch_event_payload(99, "/app"), 0)


class TestRemoveWatch:
    def test_remove_offline(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb)
        assert client.remove_watch("/app", cb) is True
        assert "/app" not in client._persistent_watchers
        assert client.remove_watch("/app", cb) is False

    def test_remove_suspended_is_local_only(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb)
        client._live.clear()
        queued = len(client._queue)
        assert client.remove_watch("/app", cb) is True
        assert len(client._queue) == queued
        assert "/app" not in client._persistent_watchers

    def test_remove_connected_queues_removewatches(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/app", cb, WatcherType.PERSISTENT)
        client._connection._write_sock = MagicMock()
        client._state = KeeperState.CONNECTED
        client._live.set()
        result = client.remove_watch_async("/app", cb)
        request, _ = client._queue[-1]
        assert isinstance(request, RemoveWatches)
        assert request.path == "/app"
        assert request.watcher_type == -1
        assert result.ready() is False  # completed by server response

    def test_shared_mode_not_removed_until_last_callback(
        self, client: KazooClient
    ) -> None:
        def cb1(event: Any) -> None: ...

        def cb2(event: Any) -> None: ...

        client.add_watch("/app", cb1, WatcherType.PERSISTENT)
        client.add_watch("/app", cb2, WatcherType.PERSISTENT)
        # While connected: removing one must not send RemoveWatches
        # because the other callback still needs the server watch.
        client._connection._write_sock = MagicMock()
        client._state = KeeperState.CONNECTED
        client._live.set()
        queued = len(client._queue)
        # Sibling callback still uses the same server watch, so no
        # RemoveWatches is sent and the result resolves to False.
        assert client.remove_watch_async("/app", cb1).get_nowait() is False
        assert len(client._queue) == queued

        result = client.remove_watch_async("/app", cb2)
        request, _ = client._queue[-1]
        assert isinstance(request, RemoveWatches)
        assert result.ready() is False  # completed by the server response

    def test_validates_arguments(self, client: KazooClient) -> None:
        def cb(event: Any) -> None: ...

        with pytest.raises(TypeError):
            client.remove_watch(123, cb)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            client.remove_watch("/app", "nope")  # type: ignore[arg-type]


class TestReRegistration:
    def _primed_requests(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> list[AddWatch]:
        connection._write_sock = MagicMock()
        connection._prime_persistent_watches()
        return [r for r, _ in client._queue if isinstance(r, AddWatch)]

    def test_snapshot_one_request_per_path_mode(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        def cb1(event: Any) -> None: ...

        def cb2(event: Any) -> None: ...

        client.add_watch("/a", cb1, WatcherType.PERSISTENT)
        client.add_watch("/a", cb2, WatcherType.PERSISTENT)
        client.add_watch("/b", cb1, WatcherType.PERSISTENT_RECURSIVE)

        requests = self._primed_requests(client, connection)
        assert [(r.path, r.watcher_type) for r in requests] == [
            ("/a", -1),
            ("/b", -2),
        ]

    def test_addwatch_precedes_queued_user_requests(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/a", cb, WatcherType.PERSISTENT)

        class UserRequest:
            type = 999

            def serialize(self) -> bytes:
                return b""

        client._queue.append((UserRequest(), client.handler.async_result()))
        connection._write_sock = MagicMock()
        connection._prime_persistent_watches()
        items = list(client._queue)
        assert isinstance(items[0][0], AddWatch)
        assert items[0][0].path == "/a"
        assert isinstance(items[1][0], UserRequest)

    def test_removed_watch_not_reregistered(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        def cb(event: Any) -> None: ...

        # Register during SUSPENDED window, then remove before reconnect.
        client.add_watch("/a", cb, WatcherType.PERSISTENT)
        client.remove_watch("/a", cb)
        requests = self._primed_requests(client, connection)
        assert requests == []

    def test_removed_watch_not_revived_after_session_loss(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        keep: list[Any] = []
        gone: list[Any] = []

        def cb_keep(event: Any) -> None:
            keep.append(event)

        def cb_gone(event: Any) -> None:
            gone.append(event)

        client.add_watch("/keep", cb_keep, WatcherType.PERSISTENT)
        client.add_watch("/gone", cb_gone, WatcherType.PERSISTENT)

        # Simulate the session-expired transition: _reset() must not
        # touch persistent watches...
        client._reset()
        assert set(client._persistent_watchers) == {"/keep", "/gone"}

        # ...user removes a watch during the recovery window...
        client.remove_watch("/gone", cb_gone)

        # ...and the fresh session primes from current local state only.
        requests = self._primed_requests(client, connection)
        assert [r.path for r in requests] == ["/keep"]

    def test_persistent_watches_survive_reset_watchers(
        self, client: KazooClient
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/a", cb, WatcherType.PERSISTENT)
        # A transient drop fires _reset_watchers (one-shot cleanup).
        client._reset_watchers()
        assert cb in client._persistent_watchers["/a"]


class TestConnectPrimesWatches:
    """_connect() must re-prime persistent watches on every
    established connection, which covers both reconnect after a
    transient drop and the new session after an expiration."""

    def _run_connect(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        # Minimal Connect deserialize result expected by _connect.
        class FakeConnectResult:
            time_out = 10000
            session_id = 1234
            protocol_version = 0
            passwd = b"\x00" * 16
            read_only = False

        sock = MagicMock()
        sock.setblocking = MagicMock()
        client.handler.create_connection = MagicMock(  # type: ignore
            return_value=sock
        )

        def fake_invoke(
            timeout: float, request: Any, xid: int | None = None
        ) -> tuple[Any, None]:
            return FakeConnectResult(), None

        connection._invoke = fake_invoke  # type: ignore[method-assign]
        connection._prime_persistent_watches = MagicMock()  # type: ignore

        read_timeout, connect_timeout = connection._connect(
            "127.0.0.1", "127.0.0.1", 2181
        )
        assert read_timeout > 0 and connect_timeout > 0
        connection._prime_persistent_watches.assert_called_once_with()

    def test_connect_primes_after_transient_drop(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        def cb(event: Any) -> None: ...

        client.add_watch("/a", cb, WatcherType.PERSISTENT)
        self._run_connect(client, connection)

    def test_connect_primes_after_session_expired(
        self, client: KazooClient, connection: ConnectionHandler
    ) -> None:
        def keep(event: Any) -> None: ...

        def dropped(event: Any) -> None: ...

        client.add_watch("/keep", keep, WatcherType.PERSISTENT)
        client.add_watch("/dropped", dropped, WatcherType.PERSISTENT)

        # Session genuinely lost: local registrations survive the reset.
        client._state = KeeperState.EXPIRED_SESSION
        client._reset()
        assert set(client._persistent_watchers) == {"/keep", "/dropped"}

        # User removes one watch while still in the recovery window.
        client.remove_watch("/dropped", dropped)

        # Use the *real* priming implementation to prove it honours
        # current local state; fake only the network/session bits.
        class FakeConnectResult:
            time_out = 10000
            session_id = 9999
            protocol_version = 0
            passwd = b"\x00" * 16
            read_only = False

        sock = MagicMock()
        sock.setblocking = MagicMock()
        client.handler.create_connection = MagicMock(  # type: ignore
            return_value=sock
        )

        def fake_invoke(
            timeout: float, request: Any, xid: int | None = None
        ) -> tuple[Any, None]:
            return FakeConnectResult(), None

        connection._invoke = fake_invoke  # type: ignore[method-assign]
        connection._write_sock = MagicMock()
        connection._connect("127.0.0.1", "127.0.0.1", 2181)

        requests = [
            r for r, _ in client._queue if isinstance(r, AddWatch)
        ]
        assert [r.path for r in requests] == ["/keep"]
