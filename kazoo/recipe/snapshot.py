"""Consistent Zookeeper Subtree Snapshot

:Maintainer: None
:Status: Unknown

:class:`ConsistentTreeReader` reads a small Zookeeper subtree (for
example a configuration tree) and only returns a result when it can
prove that the returned tree existed, in its entirety, at a single
point in time.

Reading a subtree node by node with plain ``get``/``get_children``
calls can silently mix states from different modification stages: a
child may be created, deleted, or rewritten while the read is in
progress.  This recipe performs a full read pass, then a verification
pass, and only returns the tree when both passes observe identical
node identities and version counters.

Consistency Model
-----------------

The guarantee relies on these Zookeeper ``Stat`` fields, all of which
are monotonic for the lifetime of a znode incarnation:

- ``version`` / ``mzxid``: bumped on every ``set_data`` of the node.
- ``cversion`` / ``pzxid``: bumped on every create/delete of a direct
  child (i.e. on every change of the children list).
- ``aversion``: bumped on every ACL change of the node.
- ``czxid``: the creation zxid; a delete followed by a re-create of
  the same path always yields a *new* ``czxid``, so node identity is
  distinguishable across delete/recreate cycles.

For every node of the tree the reader records ``(data, version,
mzxid, czxid, aversion)`` from a single ``get`` call and ``(children,
cversion, pzxid)`` from a single ``get_children`` call.  After the
whole tree has been read, every node's stat is read again.  Because
the counters are monotonic, equal stats in both passes prove the node
did not change between its first and second observation; since the
first pass completes entirely before the second pass begins, there is
a common interval in which *every* node of the tree matched the
recorded state.  Any mismatch -- changed data, added or removed
children, a deleted or re-created node, a changed ACL -- aborts the
attempt.

What this is **not**: this is not a transactional snapshot across
arbitrary external systems, nor across multiple Zookeeper ensembles,
and it says nothing about znodes outside the requested subtree.  It is
a best-effort, single-ensemble consistency check: if the tree keeps
changing faster than the bounded retries can keep up with, the reader
fails explicitly with :class:`UnstableTreeError` instead of returning
a "last attempt looked close enough" result.

Example Use-Case
----------------

- Loading a configuration subtree where a half-applied update must
  never be observed.

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Dict, List

from kazoo.exceptions import KazooException, NoNodeError
from kazoo.protocol.states import ZnodeStat

if TYPE_CHECKING:
    from kazoo.client import KazooClient


#: Stat fields recorded by the ``get`` call and re-checked during the
#: verification pass.  All of them are monotonic for the lifetime of a
#: znode incarnation, so equality across the two passes proves the
#: node was untouched in between.
_DATA_STAT_FIELDS = ("czxid", "mzxid", "version", "aversion")

#: Stat fields recorded by the ``get_children`` call and re-checked
#: during the verification pass; they version the children list.
_CHILDREN_STAT_FIELDS = ("cversion", "pzxid")


class TreeSnapshotError(KazooException):
    """Base class for subtree snapshot failures."""


class UnstableTreeError(TreeSnapshotError):
    """The subtree kept changing; no consistent snapshot could be
    proven within the allowed number of retries.

    .. attribute:: path

        Root path of the subtree that was being read.

    .. attribute:: attempts

        Total number of read attempts that were made.

    The exception that aborted the last attempt is available as
    ``__cause__``.
    """

    def __init__(self, path: str, attempts: int) -> None:
        super().__init__(
            "Subtree rooted at %r did not stabilize after %d attempt(s)"
            % (path, attempts)
        )
        self.path = path
        self.attempts = attempts


class TreeTooLargeError(TreeSnapshotError):
    """The subtree exceeds the configured node limit.

    .. attribute:: path

        Root path of the subtree that was being read.

    .. attribute:: max_nodes

        The node limit that was exceeded.
    """

    def __init__(self, path: str, max_nodes: int) -> None:
        super().__init__(
            "Subtree rooted at %r exceeds the limit of %d node(s)"
            % (path, max_nodes)
        )
        self.path = path
        self.max_nodes = max_nodes


class _InconsistentRead(KazooException):
    """Internal retryable error: a change was detected while reading."""


class SnapshotNode:
    """One node of a consistent subtree snapshot.

    .. attribute:: path

        Full path of the node.

    .. attribute:: data

        The node data as returned by ``get``.

    .. attribute:: stat

        The :class:`~kazoo.protocol.states.ZnodeStat` of the node as
        observed by the ``get`` call of the read pass.

    .. attribute:: children

        Dict mapping child names to :class:`SnapshotNode` objects.
    """

    __slots__ = ("path", "data", "stat", "children", "_children_stat")

    def __init__(
        self,
        path: str,
        data: bytes,
        stat: ZnodeStat,
        children_stat: ZnodeStat,
    ) -> None:
        self.path = path
        self.data = data
        self.stat = stat
        self.children: Dict[str, "SnapshotNode"] = {}
        # Stat observed by the get_children call; its cversion/pzxid
        # version the children list recorded on this node.
        self._children_stat = children_stat

    def child(self, name: str) -> "SnapshotNode":
        """Return the child :class:`SnapshotNode` called ``name``."""
        return self.children[name]

    def __repr__(self) -> str:
        return "SnapshotNode(path=%r, children=%d)" % (
            self.path,
            len(self.children),
        )


class TreeSnapshot:
    """A verified consistent snapshot of a subtree.

    .. attribute:: root

        The :class:`SnapshotNode` at the root of the subtree.

    .. attribute:: attempts

        How many read attempts were needed to obtain this snapshot.
        ``1`` means the tree was stable on the first try.
    """

    __slots__ = ("root", "attempts")

    def __init__(self, root: SnapshotNode, attempts: int) -> None:
        self.root = root
        self.attempts = attempts

    def __repr__(self) -> str:
        return "TreeSnapshot(root=%r, attempts=%d)" % (
            self.root.path,
            self.attempts,
        )


class ConsistentTreeReader:
    """Reads a small Zookeeper subtree with consistency verification.

    :param client: A :class:`~kazoo.client.KazooClient` instance.
    :param max_nodes: Maximum number of nodes the subtree may contain;
        larger trees fail with :class:`TreeTooLargeError`.
    :param max_retries: How many times a read is retried after a
        concurrent modification is detected.  The total number of
        attempts is ``1 + max_retries``.
    :param retry_delay: Base delay in seconds between attempts.
    :param backoff: Multiplier applied to the delay after each retry.
    :param sleep_func: Function used to sleep between attempts;
        defaults to :func:`time.sleep`.
    """

    def __init__(
        self,
        client: "KazooClient",
        max_nodes: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        backoff: float = 2.0,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.max_nodes = max_nodes
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff = backoff
        self.sleep_func = sleep_func

    def read(self, path: str) -> TreeSnapshot:
        """Read the subtree rooted at ``path`` consistently.

        :returns: A :class:`TreeSnapshot` whose contents provably
            matched the full subtree state at a single point in time.

        :raises:
            :exc:`~kazoo.exceptions.NoNodeError` if the root node does
            not exist.

            :exc:`~kazoo.exceptions.NoAuthError` (or any other
            Zookeeper error) if the server rejects a read; such errors
            are never swallowed or retried.

            :exc:`TreeTooLargeError` if the subtree exceeds
            ``max_nodes``.

            :exc:`UnstableTreeError` if no consistent snapshot could
            be proven within ``1 + max_retries`` attempts.
        """
        retries = 0
        delay = self.retry_delay
        while True:
            try:
                root = self._read_pass(
                    path, path, [self.max_nodes], is_root=True
                )
                self._verify_pass(root)
            except _InconsistentRead as exc:
                retries += 1
                if retries > self.max_retries:
                    raise UnstableTreeError(path, retries) from exc
                if delay > 0:
                    self.sleep_func(delay)
                delay *= self.backoff
            else:
                return TreeSnapshot(root, retries + 1)

    def _read_pass(
        self,
        root_path: str,
        path: str,
        budget: List[int],
        is_root: bool = False,
    ) -> SnapshotNode:
        """Read the whole subtree, recording data, node identity and
        children-list versions for every node."""
        if budget[0] < 1:
            raise TreeTooLargeError(root_path, self.max_nodes)
        budget[0] -= 1
        try:
            data, stat = self.client.get(path)
            children_names, children_stat = self.client.get_children(
                path, include_data=True
            )
        except NoNodeError:
            if is_root:
                raise
            raise _InconsistentRead(
                "Node %r disappeared while reading" % (path,)
            )
        node = SnapshotNode(path, data, stat, children_stat)
        prefix = path.rstrip("/")
        for name in children_names:
            child = self._read_pass(root_path, prefix + "/" + name, budget)
            node.children[name] = child
        return node

    def _verify_pass(self, node: SnapshotNode) -> None:
        """Re-read the stat of every node; any difference proves the
        read pass spanned a modification."""
        stat = self.client.exists(node.path)
        if stat is None:
            raise _InconsistentRead(
                "Node %r was deleted after it was read" % (node.path,)
            )
        for field in _DATA_STAT_FIELDS:
            if getattr(stat, field) != getattr(node.stat, field):
                raise _InconsistentRead(
                    "Node %r changed (%s) while reading" % (node.path, field)
                )
        for field in _CHILDREN_STAT_FIELDS:
            if getattr(stat, field) != getattr(node._children_stat, field):
                raise _InconsistentRead(
                    "Children of %r changed while reading" % (node.path,)
                )
        for child in node.children.values():
            self._verify_pass(child)


def read_tree(
    client: "KazooClient",
    path: str,
    max_nodes: int = 1024,
    max_retries: int = 3,
    retry_delay: float = 0.1,
    backoff: float = 2.0,
    sleep_func: Callable[[float], None] = time.sleep,
) -> TreeSnapshot:
    """Convenience wrapper around :class:`ConsistentTreeReader`.

    Takes the same parameters and returns the same
    :class:`TreeSnapshot`.
    """
    reader = ConsistentTreeReader(
        client,
        max_nodes=max_nodes,
        max_retries=max_retries,
        retry_delay=retry_delay,
        backoff=backoff,
        sleep_func=sleep_func,
    )
    return reader.read(path)
