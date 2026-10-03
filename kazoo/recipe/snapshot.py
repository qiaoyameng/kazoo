"""Read a whole ZooKeeper subtree as one verified-consistent snapshot.

Reading a configuration subtree with plain
:meth:`~kazoo.client.KazooClient.get` calls one node at a time can mix
values written by different modification stages: each individual
answer is correct for *some* point in time, but nothing ties them
together.  This module provides :func:`get_tree_snapshot`, which reads
a small subtree and only returns it when it can prove that every
returned value was live during one common stable interval.

How consistency is verified
===========================

ZooKeeper maintains a small set of monotonically increasing,
per-znode fields in every :class:`~kazoo.protocol.states.ZnodeStat`.
The snapshot protocol relies on these fields and on the server
guarantee that they never decrease and are bumped on every relevant
change:

* ``czxid`` -- the zxid that created the znode.  This is the node's
  *identity*: deleting a path and re-creating it with the same name
  and the same data still yields a fresh ``czxid``, so a recycled name
  can never be confused with the original node.
* ``mzxid`` / ``version`` -- bumped on every data change, so any
  ``set`` between two reads is visible.
* ``cversion`` / ``pzxid`` -- bumped whenever a child is added or
  removed, so any change to the child list is visible.
* ``aversion`` -- bumped on ACL changes.

Each attempt runs in two phases:

1. **Read phase** -- the tree is walked depth-first.  For every node
   both ``get`` (data + stat) and ``get_children`` (with
   ``include_data=True``, yielding the child list plus the node's
   stat) are issued, and the two stats must agree; the node's identity
   tuple is recorded.  A child that vanishes mid-walk surfaces as
   :exc:`~kazoo.exceptions.NoNodeError` and invalidates the attempt.
2. **Verify phase** -- every recorded node is re-stated via ``exists``
   and its identity tuple must match the recorded one exactly.

Because *all* reads complete before *any* verification starts, a fully
passing verification proves that a common stable interval existed --
the window between the end of the read phase and the start of the
verify phase -- during which every returned datum and child list was
simultaneously the live state.  Anything less (for example a tree that
merely "looks the same" on a later attempt while versions moved
underneath) is rejected.

Scope of the guarantee
======================

The guarantee covers exactly one ZooKeeper subtree as served by the
connected ensemble, observed through one client.  It is **not** a
transactional snapshot across ZooKeeper and any external system
(databases, files, other services), and it does not prevent future
changes: the tree may be modified again the moment the snapshot is
returned.  Callers that need to notice later changes should combine
this with watches.

Failure handling
================

* :exc:`~kazoo.exceptions.NoNodeError` raised while walking (a node
  disappeared mid-read) invalidates the attempt and is retried.
* Any other exception -- notably
  :exc:`~kazoo.exceptions.NoAuthError` for permission problems -- is
  re-raised to the caller unchanged; it is never swallowed or
  converted into a "close enough" result.
* If no attempt within ``max_retries`` verifies cleanly,
  :exc:`~kazoo.exceptions.SnapshotUnstableError` is raised.  A
  snapshot is never returned without a passed verification.
* If the tree exceeds ``max_nodes``,
  :exc:`~kazoo.exceptions.TreeTooLargeError` is raised immediately.

.. versionadded:: 2.12.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kazoo.exceptions import (
    NoNodeError,
    SnapshotUnstableError,
    TreeTooLargeError,
)
from kazoo.protocol.states import ZnodeStat

if TYPE_CHECKING:
    from kazoo.client import KazooClient


class _UnstableTreeError(Exception):
    """Internal signal: the tree changed during the current attempt."""


def _stat_identity(stat: ZnodeStat) -> tuple[int, ...]:
    """The stat fields that together pin a node's state and identity.

    Equality of this tuple at two points in time proves the node was
    not created/deleted/recreated, modified, re-ACL'd, and did not
    gain or lose children in between.
    """
    return (
        stat.czxid,
        stat.mzxid,
        stat.version,
        stat.cversion,
        stat.pzxid,
        stat.aversion,
    )


@dataclass
class TreeSnapshot:
    """One node of a verified-consistent subtree snapshot.

    .. attribute:: path

        Full path of this node.

    .. attribute:: data

        The node's data as of the snapshot's stable interval.

    .. attribute:: stat

        The :class:`~kazoo.protocol.states.ZnodeStat` captured with
        the data.

    .. attribute:: children

        Mapping of child name to :class:`TreeSnapshot`, recursively
        covering the whole subtree.
    """

    path: str
    data: bytes
    stat: ZnodeStat
    children: dict[str, "TreeSnapshot"] = field(default_factory=dict)


def _child_path(path: str, name: str) -> str:
    return path.rstrip("/") + "/" + name


def _read_attempt(
    client: KazooClient, path: str, max_nodes: int
) -> TreeSnapshot:
    """Read the whole tree once, then verify every node is unchanged.

    Raises _UnstableTreeError or NoNodeError if the tree moved during
    the attempt; any other exception propagates to the caller.
    """
    identities: dict[str, tuple[int, ...]] = {}
    node_count = 0

    def read_node(node_path: str) -> TreeSnapshot:
        nonlocal node_count
        node_count += 1
        if node_count > max_nodes:
            raise TreeTooLargeError(
                "Subtree at %r exceeds the limit of %d nodes"
                % (path, max_nodes)
            )
        data, data_stat = client.get(node_path)
        children, children_stat = client.get_children(
            node_path, include_data=True
        )
        identity = _stat_identity(data_stat)
        if identity != _stat_identity(children_stat):
            # The node changed between the two reads of this node.
            raise _UnstableTreeError(node_path)
        identities[node_path] = identity
        snapshot_children = {}
        for name in sorted(children):
            snapshot_children[name] = read_node(_child_path(node_path, name))
        return TreeSnapshot(
            path=node_path,
            data=data,
            stat=data_stat,
            children=snapshot_children,
        )

    tree = read_node(path)

    # Verify phase: every node read above must still present the exact
    # identity captured during the read phase.  Since all reads
    # finished before this phase started, a clean pass proves a common
    # stable interval for the whole tree.
    for node_path, identity in identities.items():
        stat = client.exists(node_path)
        if stat is None or _stat_identity(stat) != identity:
            raise _UnstableTreeError(node_path)
    return tree


def get_tree_snapshot(
    client: KazooClient,
    path: str,
    max_retries: int = 3,
    max_nodes: int = 1024,
) -> TreeSnapshot:
    """Read the subtree rooted at `path` as one consistent snapshot.

    The whole subtree is read and then verified against the ZooKeeper
    version fields (see the module docstring).  Only a tree whose
    verification proves a common stable interval is returned.

    :param client: A connected :class:`~kazoo.client.KazooClient`.
    :param path: Root path of the subtree to read.
    :param max_retries: How many times a failed (unstable) attempt is
                        retried before giving up.  The total number of
                        attempts is therefore ``max_retries + 1``.
    :param max_nodes: Maximum number of nodes the subtree may contain;
                      larger trees fail fast with
                      :exc:`~kazoo.exceptions.TreeTooLargeError`.
    :returns: The root :class:`TreeSnapshot`.
    :raises ~kazoo.exceptions.SnapshotUnstableError:
        If the subtree kept changing and no attempt within the retry
        limit could be verified.
    :raises ~kazoo.exceptions.TreeTooLargeError:
        If the subtree exceeds `max_nodes`.
    :raises ~kazoo.exceptions.NoAuthError:
        If the client lacks permission to read part of the subtree.
        Any other non-consistency error is likewise propagated
        unchanged.
    """
    last_error: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            return _read_attempt(client, path, max_nodes)
        except (NoNodeError, _UnstableTreeError) as exc:
            # The tree moved while it was being read; the attempt is
            # void and must not be returned.  Retry within the limit.
            last_error = exc
    raise SnapshotUnstableError(
        "Subtree at %r did not stabilize after %d attempt(s)"
        % (path, max_retries + 1)
    ) from last_error
