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

    A variant of :class:`DoubleBarrier` for a fixed group of workers
    that synchronize at the start and end of *repeated* rounds of
    work. Unlike the one-shot :class:`DoubleBarrier`, every
    participation is bound to a round and only ever touches that
    round's znodes, so a participant that is slow to leave round N
    can neither delete the ``ready`` znode of round N+1 nor be
    counted towards its quorum.

    Layout::

        <path>/rounds/<round>/members/<member>   (ephemeral)
        <path>/rounds/<round>/ready

    Round ids come from a local, monotonically increasing counter:
    every call to :meth:`enter` joins the next round. All members of
    the group must therefore call :meth:`enter` and :meth:`leave` the
    same number of times, which a typical worker loop does naturally.

    Quorum rule: a round is released once ``num_clients`` ephemeral
    member znodes exist in that round's ``members`` znode. The member
    that observes the count creates the round's ``ready`` znode;
    members that arrived earlier unblock when ``ready`` appears.

    Leave rule: a member deletes only its own member znode of the
    round it entered and blocks until that round's ``members`` znode
    is empty. The last leaver(s) remove the round's ``ready``,
    ``members`` and round znodes; concurrent removals of the same
    znodes by several leavers are harmless.

    Failure handling:

    * Process crash: the member's ephemeral znode is removed by
      ZooKeeper, so the remaining members can finish entering or
      leaving the round.
    * Lost create response: the member znode name is deterministic
      for the lifetime of this object, so a retry after a lost
      response (e.g. connection loss) finds its own znode via
      :exc:`~kazoo.exceptions.NodeExistsError` and continues without
      creating a duplicate or being counted twice.
    * Session loss: not retried (consistent with the rest of kazoo).
      :meth:`enter`/:meth:`leave` abort, the member's own znode for
      the round is removed best-effort and :attr:`participating` is
      ``False``. Register a session listener and re-enter; re-entering
      joins a new round, so the group must restart rounds together.

    Failed rounds: a round that never reaches quorum never creates
    its ``ready`` znode; members already waiting stay blocked until
    quorum is reached or their own connection/session fails. A round
    in which a member fails to *leave* is still cleaned up by the
    remaining leavers, because member znodes are ephemeral. Leftover
    znodes of an abandoned round are inert: they live under that
    round's own znode and are never counted towards a later round.

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
        """Create a Reusable Double Barrier

        :param client: A :class:`~kazoo.client.KazooClient` instance.
        :param path: The barrier path to use.
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
        self.node_name = uuid.uuid4().hex
        self._round = 0
        self._participating_round: int | None = None

    def _round_path(self, round_id: int) -> str:
        return "%s/rounds/%020d" % (self.path, round_id)

    def _members_path(self, round_id: int) -> str:
        return self._round_path(round_id) + "/members"

    def _member_path(self, round_id: int) -> str:
        return self._members_path(round_id) + "/" + self.node_name

    def _ready_path(self, round_id: int) -> str:
        return self._round_path(round_id) + "/ready"

    def enter(self) -> None:
        """Enter the barrier for the next round, blocks until all
        nodes have entered that round"""
        round_id = self._round
        self._round += 1
        try:
            self.client.retry(self._inner_enter, round_id)
            self.participating = True
            self._participating_round = round_id
        except KazooException:
            # We failed to enter, best effort cleanup of this round
            self._best_effort_cleanup(round_id)
            self.participating = False

    def _inner_enter(self, round_id: int) -> Literal[True]:
        members_path = self._members_path(round_id)
        ready_path = self._ready_path(round_id)

        # make sure this round's member parent node exists
        self.client.ensure_path(members_path)

        ready = self.client.handler.event_object()

        try:
            self.client.create(
                self._member_path(round_id),
                self._identifier.encode("utf-8"),
                ephemeral=True,
            )
        except NodeExistsError:
            # A previous create succeeded but its response was lost
            # and this is the retry of it; our deterministic node
            # name makes the retry a no-op.
            pass

        def created(event: WatchedEvent) -> None:
            if event.type == EventType.CREATED:
                ready.set()

        if self.client.exists(ready_path, watch=created):
            return True

        children = self.client.get_children(members_path)

        if len(children) < self.num_clients:
            ready.wait()
        else:
            try:
                self.client.create(ready_path)
            except NodeExistsError:
                pass
        return True

    def leave(self) -> None:
        """Leave the round last entered, blocks until all nodes have
        left that round. Does nothing if not participating."""
        if not self.participating or self._participating_round is None:
            return
        round_id = self._participating_round
        try:
            self.client.retry(self._inner_leave, round_id)
        except KazooException:  # pragma: nocover
            # Failed to cleanly leave
            self._best_effort_cleanup(round_id)
        self.participating = False
        self._participating_round = None

    def _inner_leave(self, round_id: int) -> bool:
        members_path = self._members_path(round_id)

        # Delete only our own node of this round
        try:
            self.client.delete(self._member_path(round_id))
        except NoNodeError:
            pass

        while True:
            gone = self.client.handler.event_object()

            def changed(event: WatchedEvent) -> None:
                gone.set()

            try:
                children = self.client.get_children(
                    members_path, watch=changed
                )
            except NoNodeError:
                children = []

            if not children:
                break

            gone.wait()

        # We are among the last to leave this round; remove its state.
        # Concurrent leavers race on the same paths, which is harmless.
        for path in (
            self._ready_path(round_id),
            members_path,
            self._round_path(round_id),
        ):
            try:
                self.client.delete(path)
            except (NoNodeError, NotEmptyError):
                pass
        return True

    def _best_effort_cleanup(self, round_id: int) -> None:
        try:
            self.client.retry(self.client.delete, self._member_path(round_id))
        except NoNodeError:
            pass
