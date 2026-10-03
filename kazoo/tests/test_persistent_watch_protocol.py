"""Wire-level tests for the persistent watch protocol additions.

These tests exercise the exact fixed protocol bytes documented by the
ZooKeeper 3.6 wire protocol and do not require a running cluster.
"""

from __future__ import annotations

import unittest

from kazoo.protocol import serialization
from kazoo.protocol.serialization import AddWatch, RemoveWatches
from kazoo.protocol.states import (
    AddWatchMode,
    EVENT_TYPE_MAP,
    EventType,
    watch_path_matches,
)


class TestAddWatchWireConstants(unittest.TestCase):
    def test_opcodes(self) -> None:
        # addWatch and removeWatches opcodes as defined by ZooKeeper.
        assert AddWatch.type == 23
        assert RemoveWatches.type == 18

    def test_watcher_type_codes(self) -> None:
        assert AddWatchMode.PERSISTENT.code == 4
        assert AddWatchMode.PERSISTENT_RECURSIVE.code == 5

    def test_mode_from_code(self) -> None:
        assert AddWatchMode.from_code(4) is AddWatchMode.PERSISTENT
        assert (
            AddWatchMode.from_code(5) is AddWatchMode.PERSISTENT_RECURSIVE
        )
        with self.assertRaises(ValueError):
            AddWatchMode.from_code(1)


class TestAddWatchSerialization(unittest.TestCase):
    def test_serialize_persistent(self) -> None:
        request = AddWatch("/app", AddWatchMode.PERSISTENT.code)
        # length-prefixed path "/app" + int mode (4)
        assert request.serialize() == b"\x00\x00\x00\x04/app" + (
            b"\x00\x00\x00\x04"
        )

    def test_serialize_persistent_recursive(self) -> None:
        request = AddWatch("/app", AddWatchMode.PERSISTENT_RECURSIVE.code)
        assert request.serialize() == b"\x00\x00\x00\x04/app" + (
            b"\x00\x00\x00\x05"
        )

    def test_response_is_empty(self) -> None:
        assert AddWatch.deserialize(b"", 0) is True


class TestRemoveWatchesSerialization(unittest.TestCase):
    def test_serialize_persistent(self) -> None:
        request = RemoveWatches(
            "/app", AddWatchMode.PERSISTENT.code
        )
        assert request.serialize() == b"\x00\x00\x00\x04/app" + (
            b"\x00\x00\x00\x04"
        )

    def test_serialize_persistent_recursive(self) -> None:
        request = RemoveWatches(
            "/app/sub", AddWatchMode.PERSISTENT_RECURSIVE.code
        )
        assert request.serialize() == b"\x00\x00\x00\x08/app/sub" + (
            b"\x00\x00\x00\x05"
        )

    def test_response_is_empty(self) -> None:
        assert RemoveWatches.deserialize(b"", 0) is True


class TestWatchPathMatching(unittest.TestCase):
    def test_exact_path(self) -> None:
        assert watch_path_matches("/app", "/app")

    def test_descendant(self) -> None:
        assert watch_path_matches("/app", "/app/child")
        assert watch_path_matches("/app", "/app/child/grand")

    def test_segment_boundary_respected(self) -> None:
        # /app must not match /apple or /application
        assert not watch_path_matches("/app", "/apple")
        assert not watch_path_matches("/app", "/application/x")

    def test_root_watch_matches_everything(self) -> None:
        assert watch_path_matches("/", "/anything")
        assert watch_path_matches("/", "/")

    def test_no_prefix_match_without_separator(self) -> None:
        assert not watch_path_matches("/a/b", "/a/bc")
        assert watch_path_matches("/a/b", "/a/b/c")

    def test_none_event_path(self) -> None:
        assert not watch_path_matches("/app", None)


class TestEventTypeMapping(unittest.TestCase):
    def test_persistent_watch_removed_is_type_seven(self) -> None:
        assert EVENT_TYPE_MAP[7] is EventType.PERSISTENT_WATCH_REMOVED


class TestSerializationModuleExports(unittest.TestCase):
    def test_requests_importable(self) -> None:
        assert serialization.AddWatch is AddWatch
        assert serialization.RemoveWatches is RemoveWatches
