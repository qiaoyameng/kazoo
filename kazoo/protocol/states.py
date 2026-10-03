"""Kazoo State and Event objects"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable, NamedTuple


# This is a (str, Enum) for backwards compatibility.
class KazooState(str, Enum):
    """High level connection state values

    States inspired by Netflix Curator.

    .. attribute:: SUSPENDED

        The connection has been lost but may be recovered. We should
        operate in a "safe mode" until then. When the connection is
        resumed, it may be discovered that the session expired. A
        client should not assume that locks are valid during this
        time.

    .. attribute:: CONNECTED

        The connection is alive and well.

    .. attribute:: LOST

        The connection has been confirmed dead. Any ephemeral nodes
        will need to be recreated upon re-establishing a connection.
        If locks were acquired or recipes using ephemeral nodes are in
        use, they can be considered lost as well.

    """

    SUSPENDED = "SUSPENDED"
    CONNECTED = "CONNECTED"
    LOST = "LOST"


# This is a (str, Enum) for backwards compatibility.
class KeeperState(str, Enum):
    """Zookeeper State

    Represents the Zookeeper state. Watch functions will receive a
    :class:`KeeperState` attribute as their state argument.

    .. attribute:: AUTH_FAILED

        Authentication has failed, this is an unrecoverable error.

    .. attribute:: CONNECTED

        Zookeeper is connected.

    .. attribute:: CONNECTED_RO

        Zookeeper is connected in read-only state.

    .. attribute:: CONNECTING

        Zookeeper is currently attempting to establish a connection.

    .. attribute:: EXPIRED_SESSION

        The prior session was invalid, all prior ephemeral nodes are
        gone.

    """

    AUTH_FAILED = "AUTH_FAILED"
    CONNECTED = "CONNECTED"
    CONNECTED_RO = "CONNECTED_RO"
    CONNECTING = "CONNECTING"
    CLOSED = "CLOSED"
    EXPIRED_SESSION = "EXPIRED_SESSION"


# This is a (str, Enum) for backwards compatibility.
class EventType(str, Enum):
    """Zookeeper Event

    Represents a Zookeeper event. Events trigger watch functions which
    will receive a :class:`EventType` attribute as their event
    argument.

    .. attribute:: CREATED

        A node has been created.

    .. attribute:: DELETED

        A node has been deleted.

    .. attribute:: CHANGED

        The data for a node has changed.

    .. attribute:: CHILD

        The children under a node have changed (a child was added or
        removed). This event does not indicate the data for a child
        node has changed, which must have its own watch established.

    .. attribute:: NONE

        The connection state has been altered.

    .. attribute:: PERSISTENT_WATCH_REMOVED

        A persistent watch has been removed from the server, either
        because the client explicitly asked for it to be removed or
        because the watch was force-removed by the server. The event is
        delivered to persistent and persistent-recursive watchers only.

    """

    CREATED = "CREATED"
    DELETED = "DELETED"
    CHANGED = "CHANGED"
    CHILD = "CHILD"
    PERSISTENT_WATCH_REMOVED = "PERSISTENT_WATCH_REMOVED"
    NONE = "NONE"


EVENT_TYPE_MAP = {
    -1: EventType.NONE,
    1: EventType.CREATED,
    2: EventType.DELETED,
    3: EventType.CHANGED,
    4: EventType.CHILD,
    7: EventType.PERSISTENT_WATCH_REMOVED,
}


# This is a (str, Enum) for backwards compatibility.
class AddWatchMode(str, Enum):
    """Persistent watch registration mode.

    Used with :meth:`~kazoo.client.KazooClient.add_watch` to select
    between a persistent watch that watches a single path and a
    persistent-recursive watch that also covers the whole subtree
    beneath it.

    The values are the ones sent on the wire by ZooKeeper and match the
    ``AddWatchMode``/``WatcherType`` enums of the server protocol.

    .. attribute:: PERSISTENT

        Watch a single path. The watch stays registered until it is
        explicitly removed, even after it has been triggered.

    .. attribute:: PERSISTENT_RECURSIVE

        Watch a path and every znode beneath it, with the same
        persistent semantics as :attr:`PERSISTENT`.

    """

    PERSISTENT = "PERSISTENT"
    PERSISTENT_RECURSIVE = "PERSISTENT_RECURSIVE"

    @property
    def code(self) -> int:
        """The integer WatcherType code sent in the AddWatch request."""
        return _ADD_WATCH_CODES[self]

    @classmethod
    def from_code(cls, code: int) -> AddWatchMode:
        """Return the mode matching an AddWatch/WatcherType wire code."""
        for mode, mode_code in _ADD_WATCH_CODES.items():
            if mode_code == code:
                return mode
        raise ValueError("Unknown persistent watch mode: %r" % (code,))


# WatcherType codes used in the AddWatch wire protocol.
_ADD_WATCH_CODES = {
    AddWatchMode.PERSISTENT: 4,
    AddWatchMode.PERSISTENT_RECURSIVE: 5,
}


def watch_path_matches(watch_path: str, event_path: str | None) -> bool:
    """Return whether an event path is covered by a persistent-recursive
    watch registered at ``watch_path``.

    Matching is performed on whole path segments so that a watch on
    ``/app`` matches ``/app`` and ``/app/sub/node`` but never
    ``/apple``.
    """
    if event_path is None:
        return False
    if watch_path == "/":
        return event_path.startswith("/")
    return event_path == watch_path or event_path.startswith(
        watch_path + "/"
    )


class WatchedEvent(NamedTuple):
    """A change on ZooKeeper that a Watcher is able to respond to.

    The :class:`WatchedEvent` includes exactly what happened, the
    current state of ZooKeeper, and the path of the node that was
    involved in the event. An instance of :class:`WatchedEvent` will be
    passed to registered watch functions.

    .. attribute:: type

        A :class:`EventType` attribute indicating the event type.

    .. attribute:: state

        A :class:`KeeperState` attribute indicating the Zookeeper
        state.

    .. attribute:: path

        The path of the node for the watch event.

    """

    type: EventType
    state: KeeperState
    path: str | None


class Callback(NamedTuple):
    """A callback that is handed to a handler for dispatch

    :param type: Type of the callback, currently is only 'watch'
    :param func: Callback function
    :param args: Argument list for the callback function

    """

    type: str
    func: Callable[..., Any]
    args: Iterable[Any]


class ZnodeStat(NamedTuple):
    """A ZnodeStat structure with convenience properties

    When getting the value of a znode from Zookeeper, the properties for
    the znode known as a "Stat structure" will be retrieved. The
    :class:`ZnodeStat` object provides access to the standard Stat
    properties and additional properties that are more readable and use
    Python time semantics (seconds since epoch instead of ms).

    .. note::

        The original Zookeeper Stat name is in parens next to the name
        when it differs from the convenience attribute. These are **not
        functions**, just attributes.

    .. attribute:: creation_transaction_id (czxid)

        The transaction id of the change that caused this znode to be
        created.

    .. attribute:: last_modified_transaction_id (mzxid)

        The transaction id of the change that last modified this znode.

    .. attribute:: created (ctime)

        The time in seconds from epoch when this znode was created.
        (ctime is in milliseconds)

    .. attribute:: last_modified (mtime)

        The time in seconds from epoch when this znode was last
        modified. (mtime is in milliseconds)

    .. attribute:: version

        The number of changes to the data of this znode.

    .. attribute:: acl_version (aversion)

        The number of changes to the ACL of this znode.

    .. attribute:: owner_session_id (ephemeralOwner)

        The session id of the owner of this znode if the znode is an
        ephemeral node. If it is not an ephemeral node, it will be
        `None`. (ephemeralOwner will be 0 if it is not ephemeral)

    .. attribute:: data_length (dataLength)

        The length of the data field of this znode.

    .. attribute:: children_count (numChildren)

        The number of children of this znode.

    """

    czxid: int
    mzxid: int
    ctime: int
    mtime: int
    version: int
    cversion: int
    aversion: int
    ephemeralOwner: int
    dataLength: int
    numChildren: int
    pzxid: int

    @property
    def acl_version(self) -> int:
        return self.aversion

    @property
    def children_version(self) -> int:
        return self.cversion

    @property
    def created(self) -> float:
        return self.ctime / 1000.0

    @property
    def last_modified(self) -> float:
        return self.mtime / 1000.0

    @property
    def owner_session_id(self) -> int | None:
        return self.ephemeralOwner or None

    @property
    def creation_transaction_id(self) -> int:
        return self.czxid

    @property
    def last_modified_transaction_id(self) -> int:
        return self.mzxid

    @property
    def data_length(self) -> int:
        return self.dataLength

    @property
    def children_count(self) -> int:
        return self.numChildren
