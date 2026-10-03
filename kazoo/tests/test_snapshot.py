from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest

from kazoo.exceptions import NoAuthError, NoNodeError
from kazoo.protocol.states import ZnodeStat
from kazoo.recipe.snapshot import (
    ConsistentTreeReader,
    TreeSnapshot,
    TreeTooLargeError,
    UnstableTreeError,
    read_tree,
)


class FakeZnode:
    def __init__(self, data: bytes, zxid: int) -> None:
        self.data = data
        self.czxid = zxid
        self.mzxid = zxid
        self.pzxid = zxid
        self.version = 0
        self.cversion = 0
        self.aversion = 0


class FakeEnsemble:
    """A minimal in-memory Zookeeper-like model.

    All mutations get a fresh, monotonically increasing zxid and bump
    the same Stat counters a real Zookeeper server would.
    """

    def __init__(self) -> None:
        self.zxid = 0
        self.nodes: Dict[str, FakeZnode] = {"/": FakeZnode(b"", 0)}
        self.denied: Set[str] = set()

    def _tick(self) -> int:
        self.zxid += 1
        return self.zxid

    @staticmethod
    def _parent(path: str) -> str:
        parent = path.rstrip("/").rsplit("/", 1)[0]
        return parent or "/"

    def create(self, path: str, data: bytes = b"") -> None:
        parent = self._parent(path)
        assert parent in self.nodes, "parent %r missing" % parent
        assert path not in self.nodes, "%r already exists" % path
        zxid = self._tick()
        self.nodes[path] = FakeZnode(data, zxid)
        parent_node = self.nodes[parent]
        parent_node.cversion += 1
        parent_node.pzxid = zxid

    def set_data(self, path: str, data: bytes) -> None:
        node = self.nodes[path]
        zxid = self._tick()
        node.data = data
        node.version += 1
        node.mzxid = zxid

    def set_acl(self, path: str) -> None:
        self.nodes[path].aversion += 1

    def delete(self, path: str) -> None:
        assert not self.children_of(path), "%r has children" % path
        zxid = self._tick()
        del self.nodes[path]
        parent_node = self.nodes[self._parent(path)]
        parent_node.cversion += 1
        parent_node.pzxid = zxid

    def delete_recursive(self, path: str) -> None:
        for child in self.children_of(path):
            self.delete_recursive(path.rstrip("/") + "/" + child)
        self.delete(path)

    def children_of(self, path: str) -> List[str]:
        prefix = path.rstrip("/") + "/"
        names: List[str] = []
        for candidate in self.nodes:
            if candidate.startswith(prefix):
                rest = candidate[len(prefix):]
                if rest and "/" not in rest:
                    names.append(rest)
        return sorted(names)

    def stat_of(self, path: str) -> ZnodeStat:
        node = self.nodes[path]
        return ZnodeStat(
            czxid=node.czxid,
            mzxid=node.mzxid,
            ctime=0,
            mtime=0,
            version=node.version,
            cversion=node.cversion,
            aversion=node.aversion,
            ephemeralOwner=0,
            dataLength=len(node.data),
            numChildren=len(self.children_of(path)),
            pzxid=node.pzxid,
        )


HookFunc = Callable[[str, str, "ScriptedClient"], None]


class ScriptedClient:
    """A scriptable stand-in for KazooClient.

    Records every call and runs ``interceptor(op, path, client)``
    between calls, so tests can inject mutations at exact points of
    the read.
    """

    def __init__(self, ensemble: FakeEnsemble) -> None:
        self.ensemble = ensemble
        self.calls: List[Tuple[str, str]] = []
        self.interceptor: Optional[HookFunc] = None

    def _hook(self, op: str, path: str) -> None:
        self.calls.append((op, path))
        if self.interceptor is not None:
            self.interceptor(op, path, self)

    def _node(self, path: str) -> FakeZnode:
        if path in self.ensemble.denied:
            raise NoAuthError("not authorized to read %r" % path)
        try:
            return self.ensemble.nodes[path]
        except KeyError:
            raise NoNodeError("%r does not exist" % path)

    def get(self, path: str) -> Tuple[bytes, ZnodeStat]:
        self._hook("get", path)
        node = self._node(path)
        return node.data, self.ensemble.stat_of(path)

    def get_children(
        self,
        path: str,
        watch: Optional[Callable[..., None]] = None,
        include_data: bool = False,
    ) -> Any:
        self._hook("get_children", path)
        self._node(path)
        names = self.ensemble.children_of(path)
        if include_data:
            return names, self.ensemble.stat_of(path)
        return names

    def exists(self, path: str) -> Optional[ZnodeStat]:
        self._hook("exists", path)
        if path in self.ensemble.denied:
            raise NoAuthError("not authorized to read %r" % path)
        if path not in self.ensemble.nodes:
            return None
        return self.ensemble.stat_of(path)


def inject_once(op: str, path: str, action: Callable[[], None]) -> HookFunc:
    """Return an interceptor running ``action`` exactly once, the
    first time ``(op, path)`` is called."""
    state = {"done": False}

    def hook(op_: str, path_: str, client: ScriptedClient) -> None:
        if not state["done"] and (op_, path) == (op, path_):
            state["done"] = True
            action()

    return hook


@pytest.fixture()
def ensemble() -> FakeEnsemble:
    ensemble = FakeEnsemble()
    ensemble.create("/app", b"root")
    ensemble.create("/app/db", b"db-v1")
    ensemble.create("/app/cache", b"cache-v1")
    ensemble.create("/app/cache/ttl", b"30")
    return ensemble


@pytest.fixture()
def client(ensemble: FakeEnsemble) -> ScriptedClient:
    return ScriptedClient(ensemble)


def make_reader(
    client: ScriptedClient, **kwargs: Any
) -> ConsistentTreeReader:
    kwargs.setdefault("retry_delay", 0)
    return ConsistentTreeReader(client, **kwargs)


def test_stable_tree_returned(client: ScriptedClient) -> None:
    snapshot = make_reader(client).read("/app")

    assert isinstance(snapshot, TreeSnapshot)
    assert snapshot.attempts == 1
    root = snapshot.root
    assert root.path == "/app"
    assert root.data == b"root"
    assert sorted(root.children) == ["cache", "db"]
    assert root.child("db").data == b"db-v1"
    assert root.child("cache").child("ttl").data == b"30"
    assert root.stat.czxid == client.ensemble.nodes["/app"].czxid


def test_read_pass_completes_before_verification(
    client: ScriptedClient,
) -> None:
    make_reader(client).read("/app")

    read_ops = [
        i
        for i, (op, _) in enumerate(client.calls)
        if op in ("get", "get_children")
    ]
    verify_ops = [
        i for i, (op, _) in enumerate(client.calls) if op == "exists"
    ]
    assert read_ops and verify_ops
    assert max(read_ops) < min(verify_ops)


def test_data_change_between_passes_retries(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    client.interceptor = inject_once(
        "exists", "/app", lambda: ensemble.set_data("/app/db", b"db-v2")
    )

    snapshot = make_reader(client).read("/app")

    assert snapshot.attempts == 2
    assert snapshot.root.child("db").data == b"db-v2"
    assert snapshot.root.child("db").stat.version == 1


def test_delete_recreate_is_distinguished(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    old_czxid = ensemble.nodes["/app/db"].czxid

    def recreate() -> None:
        ensemble.delete("/app/db")
        ensemble.create("/app/db", b"db-v1")  # same data as before

    client.interceptor = inject_once("exists", "/app", recreate)

    snapshot = make_reader(client).read("/app")

    assert snapshot.attempts == 2
    node = snapshot.root.child("db")
    assert node.data == b"db-v1"
    assert node.stat.czxid != old_czxid
    assert node.stat.czxid == ensemble.nodes["/app/db"].czxid


def test_new_child_detected(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    client.interceptor = inject_once(
        "exists", "/app", lambda: ensemble.create("/app/new", b"n")
    )

    snapshot = make_reader(client).read("/app")

    assert snapshot.attempts == 2
    assert snapshot.root.child("new").data == b"n"


def test_removed_child_detected(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    client.interceptor = inject_once(
        "exists", "/app", lambda: ensemble.delete_recursive("/app/cache")
    )

    snapshot = make_reader(client).read("/app")

    assert snapshot.attempts == 2
    assert sorted(snapshot.root.children) == ["db"]


def test_acl_change_detected(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    client.interceptor = inject_once(
        "exists", "/app", lambda: ensemble.set_acl("/app/db")
    )

    snapshot = make_reader(client).read("/app")

    assert snapshot.attempts == 2


def test_child_disappearing_mid_read_retries(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    # The child is deleted after /app's children were listed but
    # before the child itself is read.
    client.interceptor = inject_once(
        "get",
        "/app/cache",
        lambda: ensemble.delete_recursive("/app/cache"),
    )

    snapshot = make_reader(client).read("/app")

    assert snapshot.attempts == 2
    assert sorted(snapshot.root.children) == ["db"]


def test_unstable_tree_fails_after_bounded_retries(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    def keep_changing(op: str, path: str, client: ScriptedClient) -> None:
        if op == "exists":
            ensemble.set_data("/app/db", b"spam")

    client.interceptor = keep_changing
    sleeps: List[float] = []
    reader = make_reader(
        client, max_retries=2, retry_delay=0.5, sleep_func=sleeps.append
    )

    with pytest.raises(UnstableTreeError) as excinfo:
        reader.read("/app")

    exc = excinfo.value
    assert exc.path == "/app"
    assert exc.attempts == 3  # 1 initial attempt + 2 retries
    assert exc.__cause__ is not None  # original inconsistency chained
    assert len(sleeps) == 2


def test_noauth_error_propagates_untouched(
    client: ScriptedClient, ensemble: FakeEnsemble
) -> None:
    ensemble.denied.add("/app/db")

    with pytest.raises(NoAuthError):
        make_reader(client).read("/app")

    # Not retried, not wrapped, not mistaken for a missing node.
    assert client.calls[-1] == ("get", "/app/db")
    assert client.calls.count(("get", "/app")) == 1
    assert not any(op == "exists" for op, _ in client.calls)


def test_node_limit_enforced(client: ScriptedClient) -> None:
    with pytest.raises(TreeTooLargeError) as excinfo:
        make_reader(client, max_nodes=2).read("/app")

    assert excinfo.value.max_nodes == 2
    assert excinfo.value.path == "/app"


def test_missing_root_raises_no_node(client: ScriptedClient) -> None:
    with pytest.raises(NoNodeError):
        make_reader(client).read("/nope")

    assert client.calls == [("get", "/nope")]


def test_read_tree_convenience_wrapper(client: ScriptedClient) -> None:
    snapshot = read_tree(client, "/app", retry_delay=0)

    assert snapshot.attempts == 1
    assert snapshot.root.child("cache").child("ttl").data == b"30"
