from __future__ import annotations

import threading

from kazoo.exceptions import (
    ConnectionLoss,
    NodeExistsError,
    NoNodeError,
    NotEmptyError,
)
from kazoo.protocol.states import EventType, KazooState, WatchedEvent
from kazoo.recipe.barrier import ReusableDoubleBarrier
from kazoo.retry import KazooRetry


class _Handler:
    """Minimal stand-in for the kazoo handler abstraction."""

    @staticmethod
    def event_object() -> threading.Event:
        return threading.Event()


class _FakeStore:
    """In-memory znode tree with one-shot watches."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        # path -> (data, session id or None for persistent nodes)
        self.nodes: dict[str, tuple[bytes, str | None]] = {"/": (b"", None)}
        self.child_watches: dict[str, list] = {}
        self.exist_watches: dict[str, list] = {}
        # log of every path ever deleted, in order
        self.deleted: list[str] = []


class FakeKazooClient:
    """In-memory stand-in for the KazooClient bits the barrier uses.

    Several clients can share one :class:`_FakeStore`; each gets its
    own session id so session expiry only removes that session's
    ephemeral nodes.
    """

    def __init__(self, store: _FakeStore, session: str):
        self._store = store
        self._session = session
        self.handler = _Handler()
        self.retry = KazooRetry(max_tries=3, delay=0.01)
        # when True, the next create succeeds but its response is lost
        self.lose_next_create_response = False

    @staticmethod
    def _parent(path: str) -> str:
        return path.rsplit("/", 1)[0] or "/"

    @staticmethod
    def _fire(
        watches: dict[str, list], path: str, event_type: EventType
    ) -> None:
        for callback in watches.pop(path, []):
            callback(
                WatchedEvent(
                    type=event_type, state=KazooState.CONNECTED, path=path
                )
            )

    def _create_locked(self, path: str, data: bytes, ephemeral: bool) -> None:
        store = self._store
        store.nodes[path] = (data, self._session if ephemeral else None)
        self._fire(store.child_watches, self._parent(path), EventType.CHILD)
        self._fire(store.exist_watches, path, EventType.CREATED)

    def ensure_path(self, path: str) -> None:
        with self._store.lock:
            current = ""
            for part in path.strip("/").split("/"):
                current += "/" + part
                if current not in self._store.nodes:
                    self._create_locked(current, b"", ephemeral=False)

    def create(
        self, path: str, data: bytes = b"", ephemeral: bool = False
    ) -> str:
        with self._store.lock:
            if self._parent(path) not in self._store.nodes:
                raise NoNodeError()
            if path in self._store.nodes:
                raise NodeExistsError()
            self._create_locked(path, data, ephemeral)
            if self.lose_next_create_response:
                self.lose_next_create_response = False
                raise ConnectionLoss("create response lost")
            return path

    def delete(self, path: str) -> None:
        store = self._store
        with store.lock:
            if path not in store.nodes:
                raise NoNodeError()
            if self._children_locked(path):
                raise NotEmptyError()
            del store.nodes[path]
            store.deleted.append(path)
            self._fire(store.exist_watches, path, EventType.DELETED)
            self._fire(store.child_watches, path, EventType.DELETED)
            self._fire(
                store.child_watches, self._parent(path), EventType.CHILD
            )

    def exists(self, path: str, watch=None):
        with self._store.lock:
            if watch is not None:
                self._store.exist_watches.setdefault(path, []).append(watch)
            return path in self._store.nodes

    def get_children(self, path: str, watch=None) -> list[str]:
        with self._store.lock:
            if path not in self._store.nodes:
                raise NoNodeError()
            if watch is not None:
                self._store.child_watches.setdefault(path, []).append(watch)
            return sorted(self._children_locked(path))

    def _children_locked(self, path: str) -> list[str]:
        prefix = "/" if path == "/" else path + "/"
        return [
            p[len(prefix):]
            for p in self._store.nodes
            if p.startswith(prefix) and "/" not in p[len(prefix):]
        ]

    def expire_session(self) -> None:
        """Simulate session loss: our ephemeral nodes vanish."""
        store = self._store
        with store.lock:
            for path, (_data, session) in list(store.nodes.items()):
                if session is not None and session == self._session:
                    del store.nodes[path]
                    store.deleted.append(path)
                    self._fire(store.exist_watches, path, EventType.DELETED)
                    self._fire(store.child_watches, path, EventType.DELETED)
                    self._fire(
                        store.child_watches,
                        self._parent(path),
                        EventType.CHILD,
                    )


def _round_path(round_id: int) -> str:
    return "/barrier/rounds/%020d" % round_id


def _wait_node(client: FakeKazooClient, path: str, present: bool) -> None:
    """Block (event driven) until the node's existence matches."""
    event = threading.Event()
    if client.exists(path, watch=lambda e: event.set()) is present:
        return
    assert event.wait(5), "timed out waiting for %s" % path


def _run(target) -> threading.Thread:
    thread = threading.Thread(target=target)
    thread.start()
    return thread


class TestReusableDoubleBarrier:
    def _barriers(self, num_clients: int = 2):
        store = _FakeStore()
        client_a = FakeKazooClient(store, "session-a")
        client_b = FakeKazooClient(store, "session-b")
        barrier_a = ReusableDoubleBarrier(
            client_a, "/barrier", num_clients, identifier="a"
        )
        barrier_b = ReusableDoubleBarrier(
            client_b, "/barrier", num_clients, identifier="b"
        )
        return store, client_a, client_b, barrier_a, barrier_b

    def test_single_member_round_trip(self) -> None:
        store = _FakeStore()
        client = FakeKazooClient(store, "session-a")
        barrier = ReusableDoubleBarrier(client, "/barrier", 1)

        for round_id in range(2):
            assert not barrier.participating
            barrier.enter()
            assert barrier.participating
            barrier.leave()
            assert not barrier.participating
            assert not client.exists(_round_path(round_id))

        # leaving without participating is a no-op
        barrier.leave()
        assert not barrier.participating

    def test_consecutive_rounds_with_slow_leaver(self) -> None:
        store, client_a, client_b, barrier_a, barrier_b = self._barriers()
        round0 = _round_path(0)
        round1 = _round_path(1)

        # Round 0: both members enter and are released.
        enter_a = _run(barrier_a.enter)
        barrier_b.enter()
        enter_a.join(5)
        assert barrier_a.participating
        assert barrier_b.participating
        assert client_b.exists(round0 + "/ready")

        # Delay A's round-0 leave, first at its own member node, then
        # at the round-0 ready znode.
        holds: dict[str, tuple[threading.Event, threading.Event]] = {}
        real_delete = client_a.delete

        def held_delete(path: str) -> None:
            if path in holds:
                held, release = holds[path]
                held.set()
                assert release.wait(5)
            real_delete(path)

        member_a0 = round0 + "/members/" + barrier_a.node_name
        holds[member_a0] = (threading.Event(), threading.Event())
        holds[round0 + "/ready"] = (threading.Event(), threading.Event())
        client_a.delete = held_delete  # type: ignore[method-assign]

        leave_a = _run(barrier_a.leave)
        leave_b = _run(barrier_b.leave)

        # A's leave is stuck before removing its member node, so B's
        # leave must not complete: the stale node still counts as a
        # round-0 member.
        assert holds[member_a0][0].wait(5)
        assert client_b.exists(member_a0)
        leave_b.join(0.1)
        assert leave_b.is_alive()

        # Let A remove its member node; B's leave then finishes the
        # round-0 cleanup while A stalls again on the ready znode.
        holds[member_a0][1].set()
        assert holds[round0 + "/ready"][0].wait(5)
        leave_b.join(5)
        assert not leave_b.is_alive()
        assert not barrier_b.participating

        # B enters round 1 while A's round-0 cleanup is still pending.
        enter_b1 = _run(barrier_b.enter)
        _wait_node(client_a, round1 + "/members/" + barrier_b.node_name, True)

        # No premature release: round 1 has a single member so far.
        assert not client_b.exists(round1 + "/ready")
        enter_b1.join(0.1)
        assert enter_b1.is_alive()

        # A's delayed round-0 cleanup finishes; it must not touch any
        # round-1 state, and round 1 must still be unreleased.
        holds[round0 + "/ready"][1].set()
        leave_a.join(5)
        assert not barrier_a.participating
        assert not any(p.startswith(round1) for p in store.deleted)
        assert not client_b.exists(round1 + "/ready")
        assert enter_b1.is_alive()

        # A joins round 1: quorum is reached exactly now.
        barrier_a.enter()
        enter_b1.join(5)
        assert barrier_a.participating
        assert barrier_b.participating
        assert client_b.exists(round1 + "/ready")

        # No duplicate or phantom members were counted.
        assert client_b.get_children(round1 + "/members") == sorted(
            [barrier_a.node_name, barrier_b.node_name]
        )

        # Both leave round 1; all round state is cleaned up exactly
        # once and nobody deletes the other member's node twice.
        leave_a1 = _run(barrier_a.leave)
        barrier_b.leave()
        leave_a1.join(5)
        assert not barrier_a.participating
        assert not barrier_b.participating
        assert not client_b.exists(round1)
        assert store.deleted.count(round0 + "/ready") == 1
        assert store.deleted.count(round1 + "/ready") == 1
        assert store.deleted.count(member_a0) == 1
        member_b0 = round0 + "/members/" + barrier_b.node_name
        assert store.deleted.count(member_b0) == 1

    def test_create_response_lost_is_retried_without_duplicate(self) -> None:
        store, client_a, client_b, barrier_a, barrier_b = self._barriers()
        round0 = _round_path(0)

        # A's first member create succeeds but the response is lost.
        client_a.lose_next_create_response = True
        enter_a = _run(barrier_a.enter)

        member_a0 = round0 + "/members/" + barrier_a.node_name
        _wait_node(client_b, member_a0, True)
        # The retry sees NodeExistsError and carries on; A is now
        # blocked waiting for the second member.
        enter_a.join(0.5)
        assert enter_a.is_alive()
        assert client_b.get_children(round0 + "/members") == [
            barrier_a.node_name
        ]
        assert not client_b.exists(round0 + "/ready")
        assert client_b.exists(member_a0)

        barrier_b.enter()
        enter_a.join(5)
        assert barrier_a.participating
        assert barrier_b.participating
        assert client_b.get_children(round0 + "/members") == sorted(
            [barrier_a.node_name, barrier_b.node_name]
        )

        leave_a = _run(barrier_a.leave)
        barrier_b.leave()
        leave_a.join(5)
        assert not client_b.exists(round0)

    def test_session_expiry_during_round(self) -> None:
        store, client_a, client_b, barrier_a, barrier_b = self._barriers()
        round0 = _round_path(0)
        round1 = _round_path(1)

        enter_a = _run(barrier_a.enter)
        barrier_b.enter()
        enter_a.join(5)
        assert barrier_a.participating

        # A's session dies: its ephemeral member node disappears.
        client_a.expire_session()
        assert not client_b.exists(round0 + "/members/" + barrier_a.node_name)

        # B can still finish leaving the round on its own.
        barrier_b.leave()
        assert not barrier_b.participating
        assert not client_b.exists(round0)

        # The crashed member is not counted in the next round.
        enter_b1 = _run(barrier_b.enter)
        _wait_node(client_a, round1 + "/members/" + barrier_b.node_name, True)
        assert not client_b.exists(round1 + "/ready")
        enter_b1.join(0.1)
        assert enter_b1.is_alive()

        # A rejoins with a fresh session; the round then completes.
        barrier_a.enter()
        enter_b1.join(5)
        assert barrier_a.participating
        assert barrier_b.participating

        leave_a1 = _run(barrier_a.leave)
        barrier_b.leave()
        leave_a1.join(5)
        assert not client_b.exists(round1)
