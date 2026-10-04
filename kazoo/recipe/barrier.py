"""Zookeeper Barriers

:Maintainer: None
:Status: Unknown

"""

from __future__ import annotations

import os
import socket
import uuid
from typing import Literal, TYPE_CHECKING

from kazoo.exceptions import (
    KazooException,
    NoNodeError,
    NodeExistsError,
    NotEmptyError,
)
from kazoo.protocol.states import EventType, WatchedEvent
from kazoo.retry import ForceRetryError

if TYPE_CHECKING:
    from kazoo.client import KazooClient


class Barrier:
    """Kazoo Barrier

    Implements a barrier to block processing of a set of nodes until
    a condition is met at which point the nodes will be allowed to
    proceed. The barrier is in place if its node exists.

    .. warning::

        The :meth:`wait` function does not handle connection loss and
        may raise :exc:`~kazoo.exceptions.ConnectionLossException` if
        the connection is lost while waiting.

    """

    def __init__(self, client: KazooClient, path: str):
        """Create a Kazoo Barrier

        :param client: A :class:`~kazoo.client.KazooClient` instance.
        :param path: The barrier path to use.

        """
        self.client = client
        self.path = path

    def create(self) -> None:
        """Establish the barrier if it doesn't exist already"""
        self.client.retry(self.client.ensure_path, self.path)

    def remove(self) -> bool:
        """Remove the barrier

        :returns: Whether the barrier actually needed to be removed.
        :rtype: bool

        """
        try:
            self.client.retry(self.client.delete, self.path)
            return True
        except NoNodeError:
            return False

    def wait(self, timeout: float | None = None) -> bool:
        """Wait on the barrier to be cleared

        :returns: True if the barrier has been cleared, otherwise
                  False.
        :rtype: bool

        """
        cleared = self.client.handler.event_object()

        def wait_for_clear(event: WatchedEvent) -> None:
            if event.type == EventType.DELETED:
                cleared.set()

        exists = self.client.exists(self.path, watch=wait_for_clear)
        if not exists:
            return True

        cleared.wait(timeout)
        return cleared.is_set()


class DoubleBarrier:
    """Kazoo Double Barrier

    Double barriers are used to synchronize the beginning and end of
    a distributed task. The barrier blocks when entering it until all
    the members have joined, and blocks when leaving until all the
    members have left.

    .. note::

        You should register a listener for session loss as the process
        will no longer be part of the barrier once the session is
        gone. Connection losses will be retried with the default retry
        policy.

    """

    def __init__(
        self,
        client: KazooClient,
        path: str,
        num_clients: int,
        identifier: str | None = None,
    ):
        """Create a Double Barrier

        :param client: A :class:`~kazoo.client.KazooClient` instance.
        :param path: The barrier path to use.
        :param num_clients: How many clients must enter the barrier to
                            proceed.
        :type num_clients: int
        :param identifier: An identifier to use for this member of the
                           barrier when participating. Defaults to the
                           hostname + process id.

        """
        self.client = client
        self.path = path
        self.num_clients = num_clients
        self._identifier = identifier or "%s-%s" % (
            socket.getfqdn(),
            os.getpid(),
        )
        self.participating = False
        self.assured_path = False
        self.node_name = uuid.uuid4().hex
        self.create_path = self.path + "/" + self.node_name

    def enter(self) -> None:
        """Enter the barrier, blocks until all nodes have entered"""
        try:
            self.client.retry(self._inner_enter)
            self.participating = True
        except KazooException:
            # We failed to enter, best effort cleanup
            self._best_effort_cleanup()
            self.participating = False

    def _inner_enter(self) -> Literal[True]:
        # make sure our barrier parent node exists
        if not self.assured_path:
            self.client.ensure_path(self.path)
            self.assured_path = True

        ready = self.client.handler.event_object()

        try:
            self.client.create(
                self.create_path,
                self._identifier.encode("utf-8"),
                ephemeral=True,
            )
        except NodeExistsError:
            pass

        def created(event: WatchedEvent) -> None:
            if event.type == EventType.CREATED:
                ready.set()

        self.client.exists(self.path + "/" + "ready", watch=created)

        children = self.client.get_children(self.path)

        if len(children) < self.num_clients:
            ready.wait()
        else:
            self.client.ensure_path(self.path + "/ready")
        return True

    def leave(self) -> None:
        """Leave the barrier, blocks until all nodes have left"""
        try:
            self.client.retry(self._inner_leave)
        except KazooException:  # pragma: nocover
            # Failed to cleanly leave
            self._best_effort_cleanup()
        self.participating = False

    def _inner_leave(self) -> bool:
        # Delete the ready node if its around
        try:
            self.client.delete(self.path + "/ready")
        except NoNodeError:
            pass

        while True:
            children = self.client.get_children(self.path)
            if not children:
                return True

            if len(children) == 1 and children[0] == self.node_name:
                self.client.delete(self.create_path)
                return True

            children.sort()

            ready = self.client.handler.event_object()

            def deleted(event: WatchedEvent) -> None:
                if event.type == EventType.DELETED:
                    ready.set()

            if self.node_name == children[0]:
                # We're first, wait on the highest to leave
                if not self.client.exists(
                    self.path + "/" + children[-1], watch=deleted
                ):
                    continue

                ready.wait()
                continue

            # Delete our node
            self.client.delete(self.create_path)

            # Wait on the first
            if not self.client.exists(
                self.path + "/" + children[0], watch=deleted
            ):
                continue

            # Wait for the lowest to be deleted
            ready.wait()

    def _best_effort_cleanup(self) -> None:
        try:
            self.client.retry(self.client.delete, self.create_path)
        except NoNodeError:
            pass


class ReusableDoubleBarrier:
    """Kazoo Reusable Double Barrier

    A double barrier that the same group of workers can safely reuse
    for multiple rounds. Unlike :class:`DoubleBarrier`, which keeps all
    of its state directly under one path and is therefore only safe for
    a single use, every participation here is bound to an explicit
    *round* and only ever touches that round's own subtree::

        {path}/rounds/{round}/members/{node}   (ephemeral member nodes)
        {path}/rounds/{round}/ready            (persistent release mark)

    Because rounds are fully namespaced, a participant that is slow to
    leave a previous round can neither delete the next round's ``ready``
    node nor be counted towards the next round's quorum.

    **Release rule**

    A round is released once at least ``num_clients`` ephemeral member
    znodes exist under ``{path}/rounds/{round}/members``. The member
    that observes the quorum creates the round's ``ready`` node; every
    member waits for ``ready`` to appear. Only member nodes of *this*
    round are counted, and only this round's ``ready`` node releases
    its members.

    **Leaving**

    :meth:`leave` only operates on the round bound by the last
    successful :meth:`enter`. It blocks until every member node of
    that round is gone; the last leaver removes ``ready`` and the
    round's directories. Calling :meth:`leave` without a matching
    :meth:`enter` is a no-op and never deletes anything.

    **Failure handling**

    * Process drop: member nodes are ephemeral. A crashed member's
      node disappears when its session expires, so the remaining
      members can finish :meth:`leave`.
    * Session invalidation: the member is no longer part of the round
      once its session is gone. Register a listener for session loss;
      afterwards re-enter (preferably with a fresh round). A leftover
      member node from the expired session is detected via its
      ephemeral owner, awaited, and recreated on re-enter.
    * Lost create response: if a retried create finds its own node
      (same identifier data and session owner), entering is treated as
      successful instead of failing.
    * Failed rounds: if :meth:`enter` raises (after retries), the
      member removes its own node best-effort and is not counted. If a
      member is lost before the quorum was reached, the round never
      releases; the survivors should abandon that round identifier and
      retry with a fresh one. Leftover state of a failed round is
      confined to its own subtree and never affects other rounds.

    .. note::

        Round identifiers must be unique per round. Reusing an
        identifier is only safe once the previous round's state has
        been fully reclaimed.

    """

    def __init__(
        self,
        client: KazooClient,
        path: str,
        num_clients: int,
        identifier: str | None = None,
    ):
        """Create a Reusable Double Barrier

        :param client: A :class:`~kazoo.client.KazooClient` instance.
        :param path: The barrier path to use. Round state is kept
                     below ``path/rounds``.
        :param num_clients: How many clients must enter a round of the
                            barrier to proceed.
        :type num_clients: int
        :param identifier: An identifier to use for this member of the
                           barrier when participating. Defaults to the
                           hostname + process id.

        """
        self.client = client
        self.path = path
        self.num_clients = num_clients
        self._identifier = identifier or "%s-%s" % (
            socket.getfqdn(),
            os.getpid(),
        )
        self.participating = False
        self.round: str | None = None
        self.node_name = uuid.uuid4().hex

    @staticmethod
    def _round_token(round: str | int) -> str:
        token = str(round)
        if not token or token in (".", "..") or "/" in token:
            raise ValueError("Invalid round identifier: %r" % (round,))
        return token

    def _round_path(self, token: str) -> str:
        return self.path + "/rounds/" + token

    def _members_path(self, token: str) -> str:
        return self._round_path(token) + "/members"

    def _ready_path(self, token: str) -> str:
        return self._round_path(token) + "/ready"

    def _create_path(self, token: str) -> str:
        return self._members_path(token) + "/" + self.node_name

    def enter(self, round: str | int) -> None:
        """Enter a round of the barrier

        Blocks until ``num_clients`` members have entered this round.

        :param round: Identifier of the round to enter. Must be unique
                      per round and a valid znode name.
        :raises ValueError: If the round identifier is invalid or this
                            barrier is still bound to a previous round
                            that has not been left yet.

        """
        token = self._round_token(round)
        if self.round is not None:
            raise ValueError(
                "Still participating in round %r, leave it first"
                % (self.round,)
            )
        # Bind the round up front so a concurrent enter is rejected
        # while this one is still blocked.
        self.round = token
        try:
            self.client.retry(self._inner_enter, token)
        except KazooException:
            # We failed to enter, best effort cleanup
            self._best_effort_cleanup(token)
            self.participating = False
            self.round = None
        else:
            self.participating = True

    def _inner_enter(self, token: str) -> Literal[True]:
        members_path = self._members_path(token)
        self.client.ensure_path(members_path)

        ready = self.client.handler.event_object()
        identifier = self._identifier.encode("utf-8")

        try:
            self.client.create(
                self._create_path(token), identifier, ephemeral=True
            )
        except NodeExistsError:
            self._handle_existing_member(token)

        def created(event: WatchedEvent) -> None:
            if event.type == EventType.CREATED:
                ready.set()

        if self.client.exists(self._ready_path(token), watch=created):
            # The round was already released, e.g. a member crashed
            # after the quorum was reached and we re-entered.
            return True

        children = self.client.get_children(members_path)
        if len(children) < self.num_clients:
            ready.wait()
        else:
            self.client.ensure_path(self._ready_path(token))
        return True

    def _handle_existing_member(self, token: str) -> None:
        # A retried create after a lost response finds its own node;
        # a node left over from an expired session must vanish first.
        create_path = self._create_path(token)
        identifier = self._identifier.encode("utf-8")
        try:
            data, stat = self.client.get(create_path)
        except NoNodeError:
            raise ForceRetryError()
        if data != identifier:
            # Genuine name collision with another member.
            raise NodeExistsError()
        session_id = None
        if self.client.client_id is not None:
            session_id = self.client.client_id[0]
        owner = getattr(stat, "ephemeralOwner", None)
        if owner is None or session_id is None or owner == session_id:
            # Our own node, the earlier create response was lost.
            return
        # Zombie node from our previous, expired session. Wait for
        # ZooKeeper to remove it, then let the retry loop recreate it.
        gone = self.client.handler.event_object()

        def deleted(event: WatchedEvent) -> None:
            if event.type == EventType.DELETED:
                gone.set()

        if self.client.exists(create_path, watch=deleted) is not None:
            gone.wait()
        raise ForceRetryError()

    def leave(self, round: str | int | None = None) -> None:
        """Leave the round bound by the last successful enter

        Blocks until all members of the round have left. Only state of
        that round is touched. A call that does not match the
        currently bound round is a no-op.

        :param round: Optional explicit round identifier; must match
                      the round bound by :meth:`enter`.

        """
        token = self.round if round is None else self._round_token(round)
        if token is None or not self.participating or token != self.round:
            return
        try:
            self.client.retry(self._inner_leave, token)
        except KazooException:  # pragma: nocover
            # Failed to cleanly leave
            self._best_effort_cleanup(token)
        self.participating = False
        self.round = None

    def _inner_leave(self, token: str) -> bool:
        members_path = self._members_path(token)
        create_path = self._create_path(token)

        while True:
            try:
                children = self.client.get_children(members_path)
            except NoNodeError:
                # The round state was already fully reclaimed.
                return True

            if not children:
                self._reclaim_round(token)
                return True

            if len(children) == 1 and children[0] == self.node_name:
                try:
                    self.client.delete(create_path)
                except NoNodeError:
                    pass
                self._reclaim_round(token)
                return True

            children.sort()

            ready = self.client.handler.event_object()

            def deleted(event: WatchedEvent) -> None:
                if event.type == EventType.DELETED:
                    ready.set()

            if self.node_name == children[0]:
                # We're first, wait on the highest to leave
                if not self.client.exists(
                    members_path + "/" + children[-1], watch=deleted
                ):
                    continue

                ready.wait()
                continue

            # Delete our node if it's still around
            if self.node_name in children:
                try:
                    self.client.delete(create_path)
                except NoNodeError:
                    pass

            # Wait on the first
            if not self.client.exists(
                members_path + "/" + children[0], watch=deleted
            ):
                continue

            # Wait for the lowest to be deleted
            ready.wait()

    def _reclaim_round(self, token: str) -> None:
        # Best effort removal of the round's state. Anything that is
        # already gone, or unexpectedly non-empty because a member of
        # the same round raced us, is left alone. Other rounds live in
        # different subtrees and can never be affected.
        for path in (
            self._ready_path(token),
            self._members_path(token),
            self._round_path(token),
        ):
            try:
                self.client.delete(path)
            except (NoNodeError, NotEmptyError):
                pass

    def _best_effort_cleanup(self, token: str) -> None:
        create_path = self._create_path(token)
        try:
            data, _ = self.client.get(create_path)
        except KazooException:
            return
        if data != self._identifier.encode("utf-8"):
            # Not our node, never delete another member's node.
            return
        try:
            self.client.retry(self.client.delete, create_path)
        except KazooException:
            pass
