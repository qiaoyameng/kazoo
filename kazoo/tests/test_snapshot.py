"""Unit tests for kazoo.recipe.snapshot using a scripted fake client.

The fake client below simulates a ZooKeeper server with correct
zxid/version semantics and lets each test script mutations (data
changes, child additions, delete+recreate) to fire between individual
read calls of the snapshot protocol.
"""

from __future__ import annotations

import pytest

from kazoo.exceptions import (
    NoAuthError,
    NoNodeError,
    SnapshotUnstableError,
    TreeTooLargeError,
)
from kazoo.protocol.states import ZnodeStat
from kazoo.recipe.snapshot import get_tree_snapshot


def _split(path):
    parent, _, name = path.rpartition("/")
    return parent or "/", name


class FakeZkClient:
    """In-memory ZooKeeper stand-in with faithful version semantics.

    ``hooks`` are callables ``hook(client, op, path)`` invoked before
    every read operation; tests use them to mutate the tree between
    reads.  ``op_log`` records every ``(op, path)`` read call so tests
    can count attempts.
    """

    def __init__(self):
        self._zxid = 0
        self._nodes = {}
        self.hooks = []
        self.op_log = []
        self._create_node("/", b"")

    # -- server-side state manipulation --------------------------------

    def _next_zxid(self):
        self._zxid += 1
        return self._zxid

    def _create_node(self, path, data):
        zxid = self._next_zxid()
        self._nodes[path] = {
            "data": data,
            "czxid": zxid,
            "mzxid": zxid,
            "version": 0,
            "cversion": 0,
            "pzxid": zxid,
            "aversion": 0,
        }
        return zxid

    def create(self, path, data=b""):
        zxid = self._create_node(path, data)
        parent, _name = _split(path)
        if parent in self._nodes:
            self._nodes[parent]["cversion"] += 1
            self._nodes[parent]["pzxid"] = zxid

    def set_data(self, path, data):
        zxid = self._next_zxid()
        node = self._nodes[path]
        node["data"] = data
        node["version"] += 1
        node["mzxid"] = zxid

    def delete(self, path):
        zxid = self._next_zxid()
        parent, _name = _split(path)
        del self._nodes[path]
        self._nodes[parent]["cversion"] += 1
        self._nodes[parent]["pzxid"] = zxid

    # -- scripting helpers ----------------------------------------------

    def mutate_once(self, op, path, mutation):
        """Run ``mutation(client)`` before the first ``op`` on ``path``."""
        fired = []

        def hook(client, seen_op, seen_path):
            if not fired and seen_op == op and seen_path == path:
                fired.append(True)
                mutation(client)

        self.hooks.append(hook)

    def call_count(self, op, path):
        return sum(1 for o, p in self.op_log if o == op and p == path)

    # -- kazoo client interface ------------------------------------------

    def _child_names(self, path):
        prefix = path.rstrip("/") + "/"
        return sorted(
            p[len(prefix):]
            for p in self._nodes
            if p.startswith(prefix) and "/" not in p[len(prefix):]
        )

    def _stat(self, path):
        node = self._nodes[path]
        return ZnodeStat(
            czxid=node["czxid"],
            mzxid=node["mzxid"],
            ctime=0,
            mtime=0,
            version=node["version"],
            cversion=node["cversion"],
            aversion=node["aversion"],
            ephemeralOwner=0,
            dataLength=len(node["data"]),
            numChildren=len(self._child_names(path)),
            pzxid=node["pzxid"],
        )

    def _before_op(self, op, path):
        self.op_log.append((op, path))
        for hook in list(self.hooks):
            hook(self, op, path)

    def get(self, path, watch=None):
        self._before_op("get", path)
        if path not in self._nodes:
            raise NoNodeError()
        return self._nodes[path]["data"], self._stat(path)

    def get_children(self, path, watch=None, include_data=False):
        self._before_op("get_children", path)
        if path not in self._nodes:
            raise NoNodeError()
        children = self._child_names(path)
        if include_data:
            return children, self._stat(path)
        return children

    def exists(self, path, watch=None):
        self._before_op("exists", path)
        if path not in self._nodes:
            return None
        return self._stat(path)


@pytest.fixture()
def client():
    fake = FakeZkClient()
    fake.create("/a", b"root-data")
    fake.create("/a/b", b"v1")
    fake.create("/a/c", b"v3")
    fake.create("/a/c/d", b"v4")
    return fake


def test_stable_tree_returned_without_retries(client):
    snapshot = get_tree_snapshot(client, "/a")

    assert snapshot.path == "/a"
    assert snapshot.data == b"root-data"
    assert set(snapshot.children) == {"b", "c"}
    assert snapshot.children["b"].data == b"v1"
    assert snapshot.children["c"].children["d"].data == b"v4"
    # A stable tree needs exactly one attempt and one verification
    # per node.
    assert client.call_count("get", "/a") == 1
    assert client.call_count("exists", "/a") == 1
    assert client.call_count("exists", "/a/c/d") == 1


def test_data_change_between_read_and_verify_triggers_retry(client):
    # Change /a/b's data after it was read, while the verify phase
    # of the first attempt is in progress.
    client.mutate_once("exists", "/a", lambda c: c.set_data("/a/b", b"v2"))

    snapshot = get_tree_snapshot(client, "/a")

    assert snapshot.children["b"].data == b"v2"
    assert client.call_count("get", "/a") == 2


def test_delete_recreate_with_same_data_is_detected(client):
    # Delete /a/b and recreate it with the *same* data during the
    # first attempt's verify phase.  The fresh czxid must give the
    # recycled name away even though the content looks identical.
    def recreate(c):
        c.delete("/a/b")
        c.create("/a/b", b"v1")

    client.mutate_once("exists", "/a", recreate)

    snapshot = get_tree_snapshot(client, "/a")

    assert snapshot.children["b"].data == b"v1"
    assert snapshot.children["b"].stat.czxid == client._nodes["/a/b"]["czxid"]
    assert client.call_count("get", "/a") == 2


def test_new_child_between_read_and_verify_triggers_retry(client):
    client.mutate_once("exists", "/a", lambda c: c.create("/a/e", b"v5"))

    snapshot = get_tree_snapshot(client, "/a")

    assert set(snapshot.children) == {"b", "c", "e"}
    assert snapshot.children["e"].data == b"v5"
    assert client.call_count("get", "/a") == 2


def test_child_vanishing_mid_read_triggers_retry(client):
    # /a/c (with its own child) disappears after the child list was
    # read but before /a/c itself is fetched.
    def remove(c):
        c.delete("/a/c/d")
        c.delete("/a/c")

    client.mutate_once("get", "/a/c", remove)

    snapshot = get_tree_snapshot(client, "/a")

    assert set(snapshot.children) == {"b"}
    assert client.call_count("get", "/a") == 2


def test_permanently_unstable_tree_fails_after_limited_retries(client):
    # Bump /a/b's version during every verify phase so no attempt can
    # ever pass verification.
    def destabilize(c, op, path):
        if op == "exists" and path == "/a/b":
            c.set_data("/a/b", b"churn")

    client.hooks.append(destabilize)

    with pytest.raises(SnapshotUnstableError):
        get_tree_snapshot(client, "/a", max_retries=2)

    assert client.call_count("get", "/a") == 3


def test_noauth_error_propagates_raw_without_retry(client):
    def deny(c, op, path):
        if op == "get" and path == "/a/b":
            raise NoAuthError()

    client.hooks.append(deny)

    with pytest.raises(NoAuthError):
        get_tree_snapshot(client, "/a")

    assert client.call_count("get", "/a") == 1


def test_node_limit_fails_fast(client):
    with pytest.raises(TreeTooLargeError):
        get_tree_snapshot(client, "/a", max_nodes=2)

    assert client.call_count("get", "/a") == 1
