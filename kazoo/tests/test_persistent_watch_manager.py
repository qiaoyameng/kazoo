"""Unit tests for :class:`PersistentWatchManager`."""

from __future__ import annotations

import threading
import unittest

from kazoo.protocol.persistent import PersistentWatchManager
from kazoo.protocol.states import (
    AddWatchMode,
    EventType,
    KeeperState,
    WatchedEvent,
)


def event(
    type: EventType,
    path: str | None = "/app",
    state: KeeperState = KeeperState.CONNECTED,
) -> WatchedEvent:
    return WatchedEvent(type, state, path)


class TestRegistration(unittest.TestCase):
    def test_first_callback_requires_request(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        assert manager.add("/app", AddWatchMode.PERSISTENT, cb) is True

    def test_second_callback_shares_registration(self) -> None:
        manager = PersistentWatchManager()
        cb1 = lambda ev: None
        cb2 = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb1)
        assert manager.add("/app", AddWatchMode.PERSISTENT, cb2) is False

    def test_modes_are_separate_registrations(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        assert manager.add("/app", AddWatchMode.PERSISTENT, cb) is True
        assert (
            manager.add(
                "/app", AddWatchMode.PERSISTENT_RECURSIVE, cb
            )
            is True
        )

    def test_same_callback_added_twice_shares_one_watch(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb)
        # Idempotent: the callback is already in the set.
        assert manager.add("/app", AddWatchMode.PERSISTENT, cb) is False


class TestRemoval(unittest.TestCase):
    def test_remove_one_of_several_callbacks_keeps_watch(self) -> None:
        manager = PersistentWatchManager()
        cb1 = lambda ev: None
        cb2 = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb1)
        manager.add("/app", AddWatchMode.PERSISTENT, cb2)
        assert manager.remove("/app", AddWatchMode.PERSISTENT, cb1) is False

    def test_remove_last_callback_tears_watch_down(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb)
        assert manager.remove("/app", AddWatchMode.PERSISTENT, cb) is True

    def test_remove_unknown_callback_is_noop(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        assert manager.remove("/app", AddWatchMode.PERSISTENT, cb) is False

    def test_re_add_after_remove_requires_new_request(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb)
        manager.remove("/app", AddWatchMode.PERSISTENT, cb)
        assert manager.add("/app", AddWatchMode.PERSISTENT, cb) is True


class TestFire(unittest.TestCase):
    def test_persistent_fires_on_exact_path(self) -> None:
        manager = PersistentWatchManager()
        calls: list[WatchedEvent] = []
        manager.add(
            "/app", AddWatchMode.PERSISTENT, calls.append
        )
        assert manager.fire(event(EventType.CHANGED, "/app")) == [
            calls.append
        ]

    def test_persistent_does_not_fire_on_sibling_segment(self) -> None:
        manager = PersistentWatchManager()
        manager.add("/app", AddWatchMode.PERSISTENT, lambda ev: None)
        assert manager.fire(event(EventType.CHANGED, "/apple")) == []

    def test_persistent_ignores_child_events(self) -> None:
        manager = PersistentWatchManager()
        manager.add("/app", AddWatchMode.PERSISTENT, lambda ev: None)
        assert manager.fire(event(EventType.CHILD, "/app")) == []

    def test_recursive_fires_for_subtree(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add(
            "/app", AddWatchMode.PERSISTENT_RECURSIVE, cb
        )
        for path in ("/app", "/app/child", "/app/child/grand"):
            assert cb in manager.fire(
                event(EventType.CHANGED, path)
            )

    def test_recursive_respects_segments(self) -> None:
        manager = PersistentWatchManager()
        manager.add(
            "/app", AddWatchMode.PERSISTENT_RECURSIVE, lambda ev: None
        )
        assert manager.fire(event(EventType.CHANGED, "/apple")) == []

    def test_recursive_receives_child_events(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/", AddWatchMode.PERSISTENT_RECURSIVE, cb)
        assert cb in manager.fire(event(EventType.CHILD, "/app"))

    def test_deduplicated_when_callback_registered_in_two_modes(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb)
        manager.add("/app", AddWatchMode.PERSISTENT_RECURSIVE, cb)
        result = manager.fire(event(EventType.CHANGED, "/app"))
        assert result == [cb]

    def test_deduplicated_across_overlapping_recursive_paths(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/", AddWatchMode.PERSISTENT_RECURSIVE, cb)
        manager.add("/app", AddWatchMode.PERSISTENT_RECURSIVE, cb)
        result = manager.fire(event(EventType.CHANGED, "/app/x"))
        assert result == [cb]

    def test_distinct_callbacks_all_fire_once(self) -> None:
        manager = PersistentWatchManager()
        cb1 = lambda ev: None
        cb2 = lambda ev: None
        manager.add("/", AddWatchMode.PERSISTENT_RECURSIVE, cb1)
        manager.add("/app", AddWatchMode.PERSISTENT_RECURSIVE, cb2)
        result = manager.fire(event(EventType.CHANGED, "/app"))
        assert sorted(result, key=id) == sorted([cb1, cb2], key=id)

    def test_deleted_event_delivered_to_persistent(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb)
        assert manager.fire(event(EventType.DELETED, "/app")) == [cb]

    def test_none_connection_event_not_delivered(self) -> None:
        manager = PersistentWatchManager()
        manager.add("/app", AddWatchMode.PERSISTENT, lambda ev: None)
        assert manager.fire(event(EventType.NONE, None)) == []


class TestWatchRemoved(unittest.TestCase):
    def test_fire_only_matches_exact_registration(self) -> None:
        manager = PersistentWatchManager()
        target = lambda ev: None
        ancestor = lambda ev: None
        manager.add(
            "/app", AddWatchMode.PERSISTENT_RECURSIVE, target
        )
        manager.add(
            "/", AddWatchMode.PERSISTENT_RECURSIVE, ancestor
        )
        result = manager.fire(
            event(
                EventType.PERSISTENT_WATCH_REMOVED, "/app"
            )
        )
        assert result == [target]

    def test_discard_path_removes_all_modes(self) -> None:
        manager = PersistentWatchManager()
        cb1 = lambda ev: None
        cb2 = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb1)
        manager.add(
            "/app", AddWatchMode.PERSISTENT_RECURSIVE, cb2
        )
        removed = manager.discard_path("/app")
        assert {r.mode for r in removed} == {
            AddWatchMode.PERSISTENT,
            AddWatchMode.PERSISTENT_RECURSIVE,
        }
        assert manager.fire(event(EventType.CHANGED, "/app")) == []


class TestSnapshot(unittest.TestCase):
    def test_registrations_snapshot_is_independent(self) -> None:
        manager = PersistentWatchManager()
        cb = lambda ev: None
        manager.add("/app", AddWatchMode.PERSISTENT, cb)
        snapshot = manager.registrations()
        assert len(snapshot) == 1
        manager.remove("/app", AddWatchMode.PERSISTENT, cb)
        # Mutating the registry does not alter the snapshot.
        assert len(snapshot) == 1


class TestConcurrency(unittest.TestCase):
    def test_concurrent_add_remove_does_not_corrupt_state(self) -> None:
        manager = PersistentWatchManager()
        stop = threading.Event()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                while not stop.is_set():
                    for index in range(20):
                        cb = lambda ev, i=index: i
                        manager.add(
                            "/p", AddWatchMode.PERSISTENT, cb
                        )
                        manager.remove(
                            "/p", AddWatchMode.PERSISTENT, cb
                        )
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        stop.wait(0.2)
        stop.set()
        for thread in threads:
            thread.join()
        assert errors == []
