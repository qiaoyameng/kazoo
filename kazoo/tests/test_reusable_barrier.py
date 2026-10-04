from __future__ import annotations

import threading
import time
import unittest
from typing import Any, Callable, cast

from kazoo.client import KazooClient
from kazoo.exceptions import (
    ConnectionLoss,
    NoChildrenForEphemeralsError,
    NoNodeError,
    NodeExistsError,
    NotEmptyError,
)
from kazoo.protocol.states import EventType, KeeperState, WatchedEvent
from kazoo.recipe.barrier import ReusableDoubleBarrier
from kazoo.retry import ForceRetryError


class _Node:
    def __init__(
        self, data: bytes = b"", ephemeral_owner: int | None = None
    ) -> None:
        self.data = data
        self.ephemeral_owner = ephemeral_owner


class _FakeStat:
    def __init__(self, ephemeral_owner: int = 0) -> None:
        self.ephemeralOwner = ephemeral_owner


class _FakeHandler:
    def event_object(self) -> threading.Event:
        return threading.Event()


WatchFunc = Callable[[WatchedEvent], None]


class _InMemoryStore:
    """In-memory znode/event stand-in mimicking ZooKeeper semantics.

    Supports the subset of operations the barriers rely on, including
    one-shot exists watches, ephemeral nodes owned by a session, and
    fault injection for lost create responses and expired sessions.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.nodes: dict[str, _Node] = {"/": _Node()}
        self.watches: dict[str, list[WatchFunc]] = {}
        self.delete_hook: (
            Callable[["_InMemoryClient", str], None] | None
        ) = None
        self._lost_create_responses = 0

    @staticmethod
    def _parent(path: str) -> str:
        return path.rsplit("/", 1)[0] or "/"

    def _children(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return [
            p[len(prefix):]
            for p in self.nodes
            if p.startswith(prefix) and "/" not in p[len(prefix):]
        ]

    def _fire(self, path: str, event_type: EventType) -> None:
        watches = self.watches.pop(path, [])
        event = WatchedEvent(
            type=event_type, state=KeeperState.CONNECTED, path=path
        )
        for watch in watches:
            watch(event)

    # -- fault injection ----------------------------------------------

    def lose_next_create_response(self) -> None:
        with self.lock:
            self._lost_create_responses += 1

    def expire_session(self, client: "_InMemoryClient") -> None:
        old_session = client.client_id[0]
        client.client_id = (old_session + 1000, b"")
        self.purge_session(old_session)

    def purge_session(self, session_id: int) -> None:
        with self.lock:
            doomed = [
                p
                for p, n in self.nodes.items()
                if n.ephemeral_owner == session_id
            ]
            for path in doomed:
                del self.nodes[path]
                self._fire(path, EventType.DELETED)

    # -- znode operations ----------------------------------------------

    def ensure_path(self, path: str) -> None:
        with self.lock:
            current = ""
            for part in path.strip("/").split("/"):
                parent = self.nodes.get(current or "/")
                if parent is not None and parent.ephemeral_owner is not None:
                    raise NoChildrenForEphemeralsError()
                current += "/" + part
                if current not in self.nodes:
                    self.nodes[current] = _Node()
                    self._fire(current, EventType.CREATED)

    def create(
        self,
        client: "_InMemoryClient",
        path: str,
        data: bytes = b"",
        ephemeral: bool = False,
    ) -> None:
        with self.lock:
            parent = self.nodes.get(self._parent(path))
            if parent is None:
                raise NoNodeError()
            if parent.ephemeral_owner is not None:
                raise NoChildrenForEphemeralsError()
            if path in self.nodes:
                raise NodeExistsError()
            owner = client.client_id[0] if ephemeral else None
            self.nodes[path] = _Node(data, owner)
            self._fire(path, EventType.CREATED)
            if self._lost_create_responses > 0:
                # The create was applied but the response never made
                # it back to the client.
                self._lost_create_responses -= 1
                raise ConnectionLoss("create response lost")

    def delete(self, path: str) -> None:
        with self.lock:
            if path not in self.nodes:
                raise NoNodeError()
            if self._children(path):
                raise NotEmptyError()
            del self.nodes[path]
            self._fire(path, EventType.DELETED)

    def exists(
        self, path: str, watch: WatchFunc | None = None
    ) -> _FakeStat | None:
        with self.lock:
            if watch is not None:
                self.watches.setdefault(path, []).append(watch)
            node = self.nodes.get(path)
            if node is None:
                return None
            return _FakeStat(node.ephemeral_owner or 0)

    def get(self, path: str) -> tuple[bytes, _FakeStat]:
        with self.lock:
            node = self.nodes.get(path)
            if node is None:
                raise NoNodeError()
            return node.data, _FakeStat(node.ephemeral_owner or 0)

    def get_children(self, path: str) -> list[str]:
        with self.lock:
            if path not in self.nodes:
                raise NoNodeError()
            return self._children(path)
class _InMemoryClient:
    """Stand-in for the KazooClient subset used by the barriers."""

    def __init__(self, store: _InMemoryStore, session_id: int) -> None:
        self._store = store
        self.handler = _FakeHandler()
        self.client_id: tuple[int, bytes] = (session_id, b"")

    def retry(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        attempts = 0
        while True:
            try:
                return func(*args, **kwargs)
            except (ConnectionLoss, ForceRetryError):
                attempts += 1
                if attempts > 1000:
                    raise

    def ensure_path(self, path: str) -> None:
        self._store.ensure_path(path)

    def create(
        self, path: str, value: bytes = b"", ephemeral: bool = False
    ) -> None:
        self._store.create(self, path, value, ephemeral)

    def delete(self, path: str) -> None:
        # The hook runs outside the store lock so tests can stall a
        # delete without blocking every other client.
        hook = self._store.delete_hook
        if hook is not None:
            hook(self, path)
        self._store.delete(path)

    def exists(
        self, path: str, watch: WatchFunc | None = None
    ) -> _FakeStat | None:
        return self._store.exists(path, watch)

    def get(self, path: str) -> tuple[bytes, _FakeStat]:
        return self._store.get(path)

    def get_children(self, path: str) -> list[str]:
        return self._store.get_children(path)


def _make_barrier(
    client: _InMemoryClient,
    path: str,
    num_clients: int,
    identifier: str | None = None,
) -> ReusableDoubleBarrier:
    return ReusableDoubleBarrier(
        cast("KazooClient", client), path, num_clients, identifier
    )
class ReusableDoubleBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _InMemoryStore()
        self.clients = [
            _InMemoryClient(self.store, session_id)
            for session_id in (1, 2, 3)
        ]
        self.errors: list[Exception] = []

    def spawn(
        self, fn: Callable[..., Any], *args: Any
    ) -> threading.Thread:
        def runner() -> None:
            try:
                fn(*args)
            except Exception as exc:
                self.errors.append(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return thread

    def join(self, *threads: threading.Thread) -> None:
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive(), "thread did not finish"

    def wait_for(
        self, predicate: Callable[[], bool], timeout: float = 5.0
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition not met within %ss" % timeout)

    def exists(self, path: str) -> bool:
        return self.clients[0].exists(path) is not None

    def member_names(self, round: str) -> list[str]:
        try:
            return sorted(
                self.clients[0].get_children("/b/rounds/%s/members" % round)
            )
        except NoNodeError:
            return []

    def test_slow_previous_round_cleanup_does_not_affect_next_round(
        self,
    ) -> None:
        c1, c2 = self.clients[0], self.clients[1]
        b1 = _make_barrier(c1, "/b", 2, identifier="one")
        b2 = _make_barrier(c2, "/b", 2, identifier="two")

        # Round 1 enters and releases.
        t1 = self.spawn(b1.enter, "1")
        t2 = self.spawn(b2.enter, "1")
        self.join(t1, t2)
        assert b1.participating and b2.participating
        assert self.exists("/b/rounds/1/ready")

        # Deliberately delay the final cleanup of round 1.
        gate = threading.Event()
        reclaim_paths = {
            "/b/rounds/1/ready",
            "/b/rounds/1/members",
            "/b/rounds/1",
        }

        def hook(client: _InMemoryClient, path: str) -> None:
            if path in reclaim_paths:
                gate.wait(10)

        self.store.delete_hook = hook

        t1 = self.spawn(b1.leave)
        t2 = self.spawn(b2.leave)
        # Both leavers removed their member nodes and now stall in
        # the gated round-1 cleanup.
        self.wait_for(lambda: self.member_names("1") == [])
        assert self.exists("/b/rounds/1/ready")

        # Round 2 starts while round 1 cleanup is still pending.
        b1_next = _make_barrier(c1, "/b", 2, identifier="one")
        b2_next = _make_barrier(c2, "/b", 2, identifier="two")
        t3 = self.spawn(b1_next.enter, "2")
        self.wait_for(lambda: len(self.member_names("2")) == 1)
        # Stale round-1 state must neither release round 2 nor count
        # towards its quorum.
        time.sleep(0.1)
        assert not self.exists("/b/rounds/2/ready")
        assert not b1_next.participating

        t4 = self.spawn(b2_next.enter, "2")
        self.join(t3, t4)
        assert b1_next.participating and b2_next.participating
        assert self.exists("/b/rounds/2/ready")
        assert len(self.member_names("2")) == 2

        # Let the delayed round-1 cleanup finish.
        gate.set()
        self.join(t1, t2)
        assert not b1.participating and not b2.participating
        assert not self.exists("/b/rounds/1")
        # The old round's cleanup did not delete round 2's nodes.
        assert self.exists("/b/rounds/2/ready")
        assert len(self.member_names("2")) == 2

        t5 = self.spawn(b1_next.leave)
        t6 = self.spawn(b2_next.leave)
        self.join(t5, t6)
        assert not self.exists("/b/rounds/2")
        assert self.errors == []

    def test_lost_create_response_is_idempotent(self) -> None:
        client = self.clients[0]
        self.store.lose_next_create_response()
        b = _make_barrier(client, "/b", 1, identifier="one")
        b.enter("1")
        assert b.participating
        # The retried create did not produce a second member node.
        assert self.member_names("1") == [b.node_name]
        b.leave()
        assert not b.participating
        assert not self.exists("/b/rounds/1")
        assert self.errors == []

    def test_expired_session_member_node_is_recreated(self) -> None:
        client = self.clients[0]
        b = _make_barrier(client, "/b", 1, identifier="one")
        # Simulate a create whose response was lost, followed by the
        # session expiring before the retry: the old session's member
        # node is still around while the client already has a new
        # session.
        client.ensure_path("/b/rounds/1/members")
        client.create(
            "/b/rounds/1/members/" + b.node_name, b"one", ephemeral=True
        )
        old_session = client.client_id[0]
        client.client_id = (old_session + 1000, b"")

        # ZooKeeper removes the expired session's nodes a moment later.
        def purge() -> None:
            time.sleep(0.1)
            self.store.purge_session(old_session)

        t = self.spawn(purge)
        b.enter("1")
        self.join(t)
        assert b.participating
        assert self.member_names("1") == [b.node_name]
        _, stat = client.get("/b/rounds/1/members/" + b.node_name)
        assert stat.ephemeralOwner == client.client_id[0]
        b.leave()
        assert not b.participating
        assert self.errors == []

    def test_crashed_member_does_not_block_remaining_leavers(
        self,
    ) -> None:
        barriers = [
            _make_barrier(client, "/b", 3, identifier=str(i))
            for i, client in enumerate(self.clients)
        ]
        threads = [self.spawn(b.enter, "1") for b in barriers]
        self.join(*threads)
        assert all(b.participating for b in barriers)

        # The third member's process drops; its session expires and
        # ZooKeeper removes its ephemeral member node.
        self.store.expire_session(self.clients[2])

        t1 = self.spawn(barriers[0].leave)
        t2 = self.spawn(barriers[1].leave)
        self.join(t1, t2)
        assert not barriers[0].participating
        assert not barriers[1].participating
        assert not self.exists("/b/rounds/1")
        assert self.errors == []

    def test_stale_member_from_previous_round_not_counted(self) -> None:
        c1, c2, c3 = self.clients
        # A leftover member node from round 1 whose owner crashed
        # before its session expired.
        c3.ensure_path("/b/rounds/1/members")
        c3.create("/b/rounds/1/members/stale", b"three", ephemeral=True)

        b1 = _make_barrier(c1, "/b", 2, identifier="one")
        b2 = _make_barrier(c2, "/b", 2, identifier="two")
        t1 = self.spawn(b1.enter, "2")
        self.wait_for(lambda: len(self.member_names("2")) == 1)
        time.sleep(0.1)
        # The stale round-1 node must not count towards round 2.
        assert not self.exists("/b/rounds/2/ready")
        assert not b1.participating

        t2 = self.spawn(b2.enter, "2")
        self.join(t1, t2)
        assert b1.participating and b2.participating
        assert len(self.member_names("2")) == 2

        # The crashed owner's session expires; its node disappears
        # without affecting round 2.
        self.store.expire_session(c3)
        assert not self.exists("/b/rounds/1/members/stale")
        assert self.exists("/b/rounds/2/ready")
        assert len(self.member_names("2")) == 2

        t3 = self.spawn(b1.leave)
        t4 = self.spawn(b2.leave)
        self.join(t3, t4)
        assert self.errors == []

    def test_enter_failure_is_not_participating(self) -> None:
        client = self.clients[0]
        client.create("/some", ephemeral=True)
        b = _make_barrier(client, "/some/path", 1)
        # the barrier cannot create children under an ephemeral node
        b.enter("1")
        assert not b.participating
        assert b.round is None
        assert not self.exists("/some/path")
        # a failed enter leaves nothing behind to clean up
        b.leave()
        assert self.errors == []

    def test_same_object_reused_across_rounds(self) -> None:
        client = self.clients[0]
        b = _make_barrier(client, "/b", 1, identifier="one")
        for token in ("1", "2", "3"):
            b.enter(token)
            assert b.participating
            assert b.round == token
            b.leave()
            assert not b.participating
            assert b.round is None
            assert not self.exists("/b/rounds/" + token)
        assert self.errors == []

    def test_leave_without_enter_is_a_noop(self) -> None:
        client = self.clients[0]
        b = _make_barrier(client, "/b", 1, identifier="one")
        b.leave()
        b.leave("1")
        assert not self.exists("/b")
        b.enter("1")
        b.leave()
        # already left, must not touch anything
        b.leave()
        assert self.errors == []

    def test_enter_while_participating_raises(self) -> None:
        b = _make_barrier(self.clients[0], "/b", 2, identifier="one")
        t = self.spawn(b.enter, "1")
        self.wait_for(lambda: len(self.member_names("1")) == 1)
        with self.assertRaises(ValueError):
            b.enter("2")
        # release the round with a second member and clean up
        b2 = _make_barrier(self.clients[1], "/b", 2, identifier="two")
        b2.enter("1")
        self.join(t)
        t1 = self.spawn(b.leave)
        t2 = self.spawn(b2.leave)
        self.join(t1, t2)
        assert self.errors == []

    def test_invalid_round_identifier(self) -> None:
        b = _make_barrier(self.clients[0], "/b", 1)
        for bad in ("", "a/b", ".", ".."):
            with self.assertRaises(ValueError):
                b.enter(bad)
        assert self.errors == []
