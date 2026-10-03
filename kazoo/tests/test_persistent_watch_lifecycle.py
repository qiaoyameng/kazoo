"""End-to-end persistent watch lifecycle tests against a scripted stub.

A tiny in-process peer speaks just enough of the ZooKeeper wire
protocol over a socketpair to drive a real
:class:`~kazoo.client.KazooClient` through connect / disconnect /
reconnect / session-expired transitions. No ZooKeeper cluster and no
real network sockets are required, and the exact request bytes sent on
the wire are recorded for assertion.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import unittest

from kazoo.client import KazooClient
from kazoo.protocol.serialization import (
    int_int_long_struct,
    int_int_struct,
    int_long_int_long_struct,
    int_struct,
    reply_header_struct,
    stat_struct,
    write_buffer,
)
from kazoo.protocol.states import (
    AddWatchMode,
    EventType,
    KazooState,
    WatchedEvent,
)
from kazoo.retry import KazooRetry

log = logging.getLogger(__name__)


def _recv_exact(sock: socket.socket, length: int) -> bytes | None:
    chunks = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _frame(payload: bytes) -> bytes:
    return int_struct.pack(len(payload)) + payload


def _reply(
    xid: int, err: int = 0, payload: bytes = b"", zxid: int = 0
) -> bytes:
    return _frame(reply_header_struct.pack(xid, zxid, err) + payload)


def _parse_string(data: bytes, offset: int) -> tuple[str, int]:
    length = int_struct.unpack_from(data, offset)[0]
    offset += 4
    if length < 0:
        return "", offset
    value = data[offset : offset + length].decode("utf-8")
    return value, offset + length


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return int_struct.pack(len(encoded)) + encoded


ADD_WATCH_OP = 23
REMOVE_WATCHES_OP = 18
CLOSE_OP = -11
PING_XID = -2
AUTH_XID = -4
WATCH_XID = -1
EXISTS_OP = 3
GET_DATA_OP = 4
GET_CHILDREN_OP = 8
GET_CHILDREN2_OP = 12


def _stat() -> bytes:
    return stat_struct.pack(
        1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 8
    )


class _SocketPairProxy:
    """Proxy around an AF_UNIX socketpair end.

    The sandbox refuses ``TCP_NODELAY`` on non-TCP sockets and blocks
    ``send`` on AF_UNIX pairs while allowing ``sendmsg``. Tolerate the
    first and route writes through ``sendmsg`` so the exact same wire
    bytes reach the scripted peer.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def setsockopt(self, level: int, option: int, value: int) -> None:
        if level == socket.IPPROTO_TCP and option == socket.TCP_NODELAY:
            return
        self._sock.setsockopt(level, option, value)

    def send(self, data: bytes, flags: int = 0) -> int:
        return self._sock.sendmsg([memoryview(data)], (), flags)

    def sendall(self, data: bytes, flags: int = 0) -> None:
        view = memoryview(data)
        while view:
            sent = self._sock.sendmsg([view], (), flags)
            view = view[sent:]

    def __getattr__(self, name: str) -> object:
        return getattr(self._sock, name)


class ScriptedZooKeeper:
    """ZooKeeper protocol peer that hands out socketpairs.

    Each time the client calls ``handler.create_connection`` it receives
    one end of a fresh socketpair; the other end is served by this
    object. Connection numbers let tests assert the exact re-registration
    sequence after drops and session expirations.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._next_session_id = 0x4100
        self.sessions: set[int] = set()
        self.requests: list[tuple[int, int, str, int]] = []
        self.forced_errors: dict[tuple[int, str], list[int]] = {}
        self.connection_count = 0
        self._server_sockets: dict[int, socket.socket] = {}
        self._threads: list[threading.Thread] = []
        self._connected: dict[int, threading.Event] = {}

    def install(self, client: KazooClient) -> None:
        def create_socket_pair() -> tuple[
            _SocketPairProxy, _SocketPairProxy
        ]:
            read_end, write_end = socket.socketpair()
            return (
                _SocketPairProxy(read_end),
                _SocketPairProxy(write_end),
            )

        def create_connection(
            *args: object, **kwargs: object
        ) -> _SocketPairProxy:
            client_end, server_end = socket.socketpair()
            client_end.setblocking(True)
            server_end.settimeout(0.5)
            with self._lock:
                self.connection_count += 1
                number = self.connection_count
                server_end_proxy = _SocketPairProxy(server_end)
                self._server_sockets[number] = server_end_proxy
                self._connected[number] = threading.Event()
            thread = threading.Thread(
                target=self._serve,
                args=(server_end_proxy, number),
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
            return _SocketPairProxy(client_end)

        client.handler.create_connection = create_connection  # type: ignore[method-assign]
        client.handler.create_socket_pair = create_socket_pair  # type: ignore[method-assign]

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sockets = list(self._server_sockets.values())
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass
        for thread in self._threads:
            thread.join(2)

    def wait_for_connection(
        self, number: int, timeout: float = 8
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                event = self._connected.get(number)
            if event is not None and event.is_set():
                return True
            time.sleep(0.02)
        return False

    def wait_for_requests(
        self, count: int, timeout: float = 8
    ) -> list[tuple[int, int, str, int]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.requests) >= count:
                    return list(self.requests)
            time.sleep(0.02)
        with self._lock:
            return list(self.requests)

    def requests_on(
        self, connection: int
    ) -> list[tuple[int, int, str, int]]:
        with self._lock:
            return [
                request
                for request in self.requests
                if request[0] == connection
            ]

    def last_session_id(self) -> int:
        with self._lock:
            return self._next_session_id - 1

    def expire_last_session(self) -> None:
        with self._lock:
            session_id = self.last_session_id()
            self.sessions.discard(session_id)
            sock = self._server_sockets.get(self.connection_count)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def drop_active_connection(self) -> None:
        with self._lock:
            sock = self._server_sockets.get(self.connection_count)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def send_watch_event(
        self, event_type: int, path: str, state: int = 3
    ) -> None:
        body = int_int_struct.pack(event_type, state) + _string(path)
        packet = _frame(
            reply_header_struct.pack(WATCH_XID, 0, 0) + body
        )
        with self._lock:
            sock = self._server_sockets.get(self.connection_count)
        assert sock is not None
        try:
            sock.sendall(packet)
        except OSError:
            pass

    def force_error(self, opcode: int, path: str, err: int) -> None:
        with self._lock:
            self.forced_errors.setdefault((opcode, path), []).append(err)

    def _serve(self, sock: socket.socket, number: int) -> None:
        try:
            # Every connection starts with exactly one Connect packet,
            # which carries no xid/opcode prefix.
            header = _recv_exact(sock, 4)
            if header is None:
                return
            payload = _recv_exact(
                sock, int_struct.unpack(header)[0]
            )
            if payload is None:
                return
            self._handle_connect(sock, payload)
            with self._lock:
                self._connected[number].set()

            while not self._stop.is_set():
                header = _recv_exact(sock, 4)
                if header is None:
                    return
                payload = _recv_exact(
                    sock, int_struct.unpack(header)[0]
                )
                if payload is None:
                    return
                self._handle_request(sock, number, payload)
        except OSError:
            return

    def _handle_request(
        self, sock: socket.socket, number: int, payload: bytes
    ) -> None:
        xid = int_struct.unpack_from(payload, 0)[0]
        offset = 4
        opcode = int_struct.unpack_from(payload, offset)[0]
        offset += 4

        if xid == PING_XID:
            sock.sendall(_reply(PING_XID))
            return
        if xid == AUTH_XID:
            sock.sendall(_reply(AUTH_XID))
            return
        if opcode == CLOSE_OP:
            return

        path, offset = _parse_string(payload, offset)
        if opcode in (ADD_WATCH_OP, REMOVE_WATCHES_OP):
            mode = int_struct.unpack_from(payload, offset)[0]
            with self._lock:
                self.requests.append((number, opcode, path, mode))
                errors = self.forced_errors.get((opcode, path))
                err = errors.pop(0) if errors else 0
            sock.sendall(_reply(xid, err))
            return

        # Minimal read responses so chroot probes and watch-bearing
        # reads (not exercised directly here) do not blow up parsing.
        if opcode == EXISTS_OP:
            sock.sendall(_reply(xid, payload=_stat()))
            return
        if opcode == GET_DATA_OP:
            sock.sendall(
                _reply(xid, payload=_string("") + _stat())
            )
            return
        if opcode == GET_CHILDREN_OP:
            sock.sendall(_reply(xid, payload=int_struct.pack(0)))
            return
        if opcode == GET_CHILDREN2_OP:
            sock.sendall(
                _reply(xid, payload=int_struct.pack(0) + _stat())
            )
            return

        sock.sendall(_reply(xid))

    def _handle_connect(self, sock: socket.socket, payload: bytes) -> None:
        proto, _zxid, timeout, session_id = (
            int_long_int_long_struct.unpack_from(payload, 0)
        )

        if session_id and session_id not in self.sessions:
            # Unknown/expired session: zero timeout forces expiry.
            body = int_int_long_struct.pack(proto, 0, 0)
            body += write_buffer(b"\x00" * 16)
            body += b"\x00"
            sock.sendall(_frame(body))
            return

        if not session_id:
            with self._lock:
                session_id = self._next_session_id
                self._next_session_id += 1
                self.sessions.add(session_id)

        body = int_int_long_struct.pack(proto, timeout, session_id)
        body += write_buffer(b"\x00" * 16)
        body += b"\x00"
        sock.sendall(_frame(body))


def _make_client() -> KazooClient:
    return KazooClient(
        hosts="127.0.0.1:2181",
        timeout=3.0,
        connection_retry=KazooRetry(
            max_tries=-1, delay=0.05, backoff=1, max_jitter=0.05
        ),
    )


def _wait_for_request(
    server: ScriptedZooKeeper,
    connection: int,
    opcode: int,
    path: str,
    mode: int,
    timeout: float = 8,
) -> bool:
    expected = (connection, opcode, path, mode)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected in server.requests_on(connection):
            return True
        time.sleep(0.02)
    return expected in server.requests_on(connection)


class PersistentWatchLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ScriptedZooKeeper()
        self.client = _make_client()
        self.server.install(self.client)
        self.client.start(timeout=8)

    def tearDown(self) -> None:
        try:
            self.client.stop()
            self.client.close()
        except Exception:
            pass
        self.server.stop()

    def test_add_watch_sends_fixed_bytes(self) -> None:
        fired = threading.Event()

        def cb(event: WatchedEvent) -> None:
            fired.set()

        result = self.client.add_watch(
            "/app", cb, AddWatchMode.PERSISTENT_RECURSIVE
        )
        assert result is True

        records = self.server.wait_for_requests(1)
        assert records == [(1, ADD_WATCH_OP, "/app", 5)]

        # Events keep firing; the watch stays registered.
        self.server.send_watch_event(3, "/app")
        assert fired.wait(5)
        fired.clear()
        self.server.send_watch_event(3, "/app/sub")
        assert fired.wait(5)

        # Joining an existing registration does not send another request.
        result = self.client.add_watch(
            "/app", lambda ev: None, AddWatchMode.PERSISTENT_RECURSIVE
        )
        assert result is False

    def test_remove_watch_sends_remove_watches(self) -> None:
        def cb(event: WatchedEvent) -> None:
            return None

        self.client.add_watch("/app", cb)
        self.server.wait_for_requests(1)

        removed = self.client.remove_watch("/app", cb)
        assert removed is True
        records = self.server.wait_for_requests(2)
        assert records[-1] == (1, REMOVE_WATCHES_OP, "/app", 4)

        # Removing again is a local no-op.
        assert self.client.remove_watch("/app", cb) is False

    def test_watch_survives_transient_disconnect(self) -> None:
        events: list[WatchedEvent] = []
        got_event = threading.Event()

        def cb(event: WatchedEvent) -> None:
            events.append(event)
            got_event.set()

        self.client.add_watch(
            "/app", cb, AddWatchMode.PERSISTENT_RECURSIVE
        )
        self.server.wait_for_requests(1)

        states: list[KazooState] = []
        reconnected = threading.Event()

        def listener(state: KazooState) -> None:
            states.append(state)
            if (
                state == KazooState.CONNECTED
                and KazooState.SUSPENDED in states
            ):
                reconnected.set()

        self.client.add_listener(listener)

        self.server.drop_active_connection()
        assert reconnected.wait(8)
        assert self.server.wait_for_connection(2)

        # The persistent watch is re-registered on the new connection.
        assert _wait_for_request(
            self.server, 2, ADD_WATCH_OP, "/app", 5
        )

        # Still wired up end to end after recovery.
        self.server.send_watch_event(3, "/app/sub")
        assert got_event.wait(5)
        assert events[-1].path == "/app/sub"

    def test_watch_reregistered_after_real_session_expiration(self) -> None:
        def cb(event: WatchedEvent) -> None:
            return None

        self.client.add_watch(
            "/app", cb, AddWatchMode.PERSISTENT
        )
        self.server.wait_for_requests(1)

        states: list[KazooState] = []
        back = threading.Event()

        def listener(state: KazooState) -> None:
            states.append(state)
            if state == KazooState.CONNECTED and KazooState.LOST in states:
                back.set()

        self.client.add_listener(listener)

        session_before = self.client.client_id[0]
        connection_before = self.server.connection_count
        self.server.expire_last_session()
        assert back.wait(8)
        # Expiry typically needs two attempts: one reuses the dead
        # session and gets a zero-timeout reply, the next establishes a
        # brand new session.
        new_connection = self.server.connection_count
        assert new_connection >= connection_before + 1

        session_after = self.client.client_id[0]
        assert session_after != session_before

        # The locally retained watch is registered against the new session.
        assert _wait_for_request(
            self.server,
            new_connection,
            ADD_WATCH_OP,
            "/app",
            4,
        )

    def test_removed_watch_not_resurrected_during_recovery(self) -> None:
        def cb(event: WatchedEvent) -> None:
            return None

        self.client.add_watch(
            "/app", cb, AddWatchMode.PERSISTENT_RECURSIVE
        )
        self.server.wait_for_requests(1)

        states: list[KazooState] = []
        back = threading.Event()

        def listener(state: KazooState) -> None:
            states.append(state)
            if (
                state == KazooState.CONNECTED
                and KazooState.SUSPENDED in states
            ):
                back.set()

        self.client.add_listener(listener)

        self.server.drop_active_connection()
        assert back.wait(8)
        assert self.server.wait_for_connection(2)

        assert self.client.remove_watch(
            "/app", cb, AddWatchMode.PERSISTENT_RECURSIVE
        )

        records = self.server.wait_for_requests(3)
        opcodes_on_two = [
            req[1] for req in records if req[0] == 2
        ]
        assert ADD_WATCH_OP in opcodes_on_two
        assert opcodes_on_two[-1] == REMOVE_WATCHES_OP

        # One more drop: nothing is left to re-register.
        back.clear()
        states.clear()
        self.server.drop_active_connection()
        assert back.wait(8)
        assert self.server.wait_for_connection(3)
        time.sleep(0.2)
        assert self.server.requests_on(3) == []

    def test_remove_while_suspended_is_honoured_on_reconnect(self) -> None:
        def cb(event: WatchedEvent) -> None:
            return None

        self.client.add_watch(
            "/cfg", cb, AddWatchMode.PERSISTENT
        )
        self.server.wait_for_requests(1)

        suspended = threading.Event()

        def listener(state: KazooState) -> None:
            if state == KazooState.SUSPENDED:
                suspended.set()

        self.client.add_listener(listener)

        self.server.drop_active_connection()
        assert suspended.wait(5)

        # Removal during the suspension window is honoured.
        self.client.remove_watch("/cfg", cb)

        assert self.server.wait_for_connection(2)
        records = self.server.requests_on(2)
        assert all(req[1] != ADD_WATCH_OP for req in records)
        assert (2, REMOVE_WATCHES_OP, "/cfg", 4) in records

    def test_persistent_watch_removed_event_cleans_local_state(
        self,
    ) -> None:
        seen: list[EventType] = []
        done = threading.Event()

        def cb(event: WatchedEvent) -> None:
            seen.append(event.type)
            done.set()

        self.client.add_watch("/app", cb)
        self.server.wait_for_requests(1)

        self.server.send_watch_event(7, "/app")
        assert done.wait(5)
        assert seen == [EventType.PERSISTENT_WATCH_REMOVED]

        # Later change events do nothing, and reconnect sends no request.
        self.server.send_watch_event(3, "/app")
        time.sleep(0.1)
        assert seen == [EventType.PERSISTENT_WATCH_REMOVED]

        back = threading.Event()

        def listener(state: KazooState) -> None:
            if state == KazooState.CONNECTED:
                back.set()

        self.client.add_listener(listener)
        self.server.drop_active_connection()
        assert back.wait(8)
        assert self.server.wait_for_connection(2)
        time.sleep(0.2)
        assert self.server.requests_on(2) == []

    def test_no_watcher_error_treats_removal_as_success(self) -> None:
        def cb(event: WatchedEvent) -> None:
            return None

        self.client.add_watch("/app", cb)
        self.server.wait_for_requests(1)

        # Server claims it never held the watcher.
        from kazoo.exceptions import NoWatcherError

        self.server.force_error(
            REMOVE_WATCHES_OP, "/app", NoWatcherError.code
        )
        result = self.client.remove_watch("/app", cb)
        assert result is False

        # Local state is gone regardless.
        time.sleep(0.1)
        self.server.drop_active_connection()
        assert self.server.wait_for_connection(2)
        time.sleep(0.2)
        assert self.server.requests_on(2) == []

    def test_stop_clears_persistent_watch_registrations(self) -> None:
        def cb(event: WatchedEvent) -> None:
            return None

        self.client.add_watch(
            "/cfg", cb, AddWatchMode.PERSISTENT_RECURSIVE
        )
        self.server.wait_for_requests(1)

        self.client.stop()
        self.client.close()
        self.server.stop()

        # The local registry is empty after an explicit stop, so even a
        # later restart would not re-register anything.
        assert not self.client._persistent_watchers

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.client.add_watch(123, lambda ev: None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.client.add_watch("/app", "not callable")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.client.add_watch("/app", lambda ev: None, mode=4)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.client.remove_watch("/app", "x")  # type: ignore[arg-type]

    def test_multiple_callbacks_share_one_server_watch(self) -> None:
        def cb1(event: WatchedEvent) -> None:
            return None

        def cb2(event: WatchedEvent) -> None:
            return None

        assert self.client.add_watch("/app", cb1) is True
        assert self.client.add_watch("/app", cb2) is False
        assert self.client.remove_watch("/app", cb1) is False
        assert self.client.remove_watch("/app", cb2) is True

        records = self.server.wait_for_requests(2)
        assert records == [
            (1, ADD_WATCH_OP, "/app", 4),
            (1, REMOVE_WATCHES_OP, "/app", 4),
        ]


class PersistentWatchChrootTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ScriptedZooKeeper()
        self.client = KazooClient(
            hosts="127.0.0.1:2181/chroot",
            timeout=3.0,
            connection_retry=KazooRetry(
                max_tries=-1,
                delay=0.05,
                backoff=1,
                max_jitter=0.05,
            ),
        )
        self.server.install(self.client)
        self.client.start(timeout=8)

    def tearDown(self) -> None:
        try:
            self.client.stop()
            self.client.close()
        except Exception:
            pass
        self.server.stop()

    def test_wire_path_is_rooted_callback_path_is_not(self) -> None:
        seen: list[str] = []
        got = threading.Event()

        def cb(event: WatchedEvent) -> None:
            seen.append(event.path)
            got.set()

        self.client.add_watch(
            "/app", cb, AddWatchMode.PERSISTENT_RECURSIVE
        )
        records = self.server.wait_for_requests(1)
        assert records == [(1, ADD_WATCH_OP, "/chroot/app", 5)]

        self.server.send_watch_event(3, "/chroot/app/sub")
        assert got.wait(5)
        assert seen == ["/app/sub"]
