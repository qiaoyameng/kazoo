from __future__ import annotations

import threading
import time
import unittest

import pytest

from kazoo.client import KazooClient
from kazoo.exceptions import (
    ConnectionLoss,
    NoNodeError,
    UnimplementedApiError,
)
from kazoo.protocol.serialization import (
    GetChildren,
    GetData,
    int_int_long_struct,
    int_struct,
    multiheader_struct,
    reply_header_struct,
    stat_struct,
    write_buffer,
    write_string,
)

MULTI_READ_OP = 22
PING_OP = 11
AUTH_OP = 100
CLOSE_OP = -11


class FakeSocket:
    """An in-memory socket substitute with two paired ends."""

    def __init__(self) -> None:
        self._in: bytearray = bytearray()
        self._peer: FakeSocket | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.closed = False

    def setblocking(self, flag: bool) -> None:
        pass

    def settimeout(self, timeout: float | None) -> None:
        pass

    def fileno(self) -> int:
        # Only used as an identity key by fake_select; never passed to
        # a real selector.
        return id(self) & 0xFFFF

    def pending(self) -> int:
        with self._lock:
            return len(self._in)

    def send(self, data: bytes) -> int:
        assert self._peer is not None
        with self._peer._cond:
            if self._peer.closed:
                raise OSError("socket closed")
            self._peer._in.extend(data)
            self._peer._cond.notify_all()
        return len(data)

    def sendall(self, data: bytes) -> None:
        sent = 0
        while sent < len(data):
            sent += self.send(data[sent:])

    def recv(self, size: int) -> bytes:
        with self._cond:
            while not self._in and not self.closed:
                self._cond.wait(0.01)
            if not self._in:
                return b""
            data = bytes(self._in[:size])
            del self._in[:size]
            return data

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()
        if self._peer is not None:
            with self._peer._cond:
                self._peer._cond.notify_all()


def socket_pair() -> tuple[FakeSocket, FakeSocket]:
    first, second = FakeSocket(), FakeSocket()
    first._peer = second
    second._peer = first
    return first, second


def _recv_exact(conn: FakeSocket, length: int) -> bytes:
    parts: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("fake server: socket closed")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _read_packet(conn: FakeSocket) -> bytes:
    (length,) = int_struct.unpack(_recv_exact(conn, 4))
    return _recv_exact(conn, length)


def _send_packet(conn: FakeSocket, payload: bytes) -> None:
    conn.sendall(int_struct.pack(len(payload)) + payload)


def _reply(xid: int, err: int, body: bytes = b"") -> bytes:
    return reply_header_struct.pack(xid, 0, err) + body


def _connect_response() -> bytes:
    return (
        int_int_long_struct.pack(0, 10000, 123456789)
        + write_buffer(b"\x00" * 16)
        + b"\x00"
    )


# Fixed-fragment builders for server responses.

def getDataResult(data: bytes) -> bytes:
    return (
        multiheader_struct.pack(GetData.type, 0, -1)
        + write_buffer(data)
        + stat_struct.pack(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    )


def getChildrenResult(children: list[str]) -> bytes:
    body = bytearray(multiheader_struct.pack(GetChildren.type, 0, -1))
    body.extend(int_struct.pack(len(children)))
    for child in children:
        body.extend(write_string(child))
    return bytes(body)


def errorResult(code: int) -> bytes:
    return multiheader_struct.pack(-1, 0, code) + int_struct.pack(code)


def terminator() -> bytes:
    return multiheader_struct.pack(-1, 1, -1)


def expected_multi_read_body(ops: list[tuple[str, int]]) -> bytes:
    body = bytearray()
    for path, op_type in ops:
        body.extend(multiheader_struct.pack(op_type, 0, -1))
        body.extend(write_string(path))
        body.extend(b"\x00")
    body.extend(multiheader_struct.pack(-1, 1, -1))
    return bytes(body)


class FakeZooKeeperServer:
    """Speaks enough of the ZK wire protocol over a FakeSocket.

    Handles Connect/Ping/Auth and hands each real request to a
    handler. Records the opcode and fixed request body of every
    request for assertions.
    """

    def __init__(self, handler) -> None:
        self._handler = handler
        self.requests: list[tuple[int, int, bytes]] = []
        self.client_sock, self._server_sock = socket_pair()
        self._wake_read, self._wake_write = socket_pair()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout)

    def close(self) -> None:
        self._server_sock.close()
        self.client_sock.close()
        self._wake_read.close()
        self._wake_write.close()

    def fake_select(self, rlist, wlist, xlist, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            readable = [s for s in rlist if s.pending() > 0 or s.closed]
            writable = [s for s in wlist if not s.closed]
            if readable or writable:
                return readable, writable, []
            if deadline is not None and time.monotonic() >= deadline:
                return [], [], []
            time.sleep(0.005)

    def _run(self) -> None:
        conn = self._server_sock
        connected = False
        while True:
            try:
                payload = _read_packet(conn)
            except (ConnectionError, OSError):
                return
            if not connected:
                _send_packet(conn, _connect_response())
                connected = True
                continue
            xid = int_struct.unpack_from(payload, 0)[0]
            op_type = int_struct.unpack_from(payload, 4)[0]
            body = payload[8:]
            if op_type == PING_OP:
                _send_packet(conn, _reply(xid, 0))
                continue
            if op_type == AUTH_OP:
                _send_packet(conn, _reply(xid, 0))
                continue
            if op_type == CLOSE_OP:
                _send_packet(conn, _reply(xid, 0))
                return
            self.requests.append((xid, op_type, body))
            _send_packet(conn, self._handler(xid, op_type, body))


class MultiReadHarness:
    """Wires a KazooClient to a FakeZooKeeperServer."""

    def __init__(self, handler, chroot: str | None = None) -> None:
        self.server = FakeZooKeeperServer(handler)
        hosts = "127.0.0.1:2181"
        if chroot:
            hosts += chroot
        self.client = KazooClient(
            hosts=hosts,
            timeout=5,
            connection_retry={
                "max_tries": 3,
                "delay": 0.01,
                "backoff": 1,
                "max_jitter": 0,
            },
            command_retry={"max_tries": 1},
        )
        client_sock = self.server.client_sock
        wake_read = self.server._wake_read
        handler = self.client.handler
        handler.select = self.server.fake_select  # type: ignore[method-assign]

        def create_connection(*args: object, **kwargs: object):
            return client_sock

        def create_socket_pair():
            return wake_read, self.server._wake_write

        handler.create_connection = (  # type: ignore[method-assign]
            create_connection
        )
        handler.create_socket_pair = (  # type: ignore[method-assign]
            create_socket_pair
        )

    def start(self) -> None:
        self.server.start()
        self.client.start(8)

    def stop(self) -> None:
        self.client.stop()
        self.server.close()
        self.server.join()


class TestMultiReadProtocol(unittest.TestCase):
    def test_empty_batch_fixed_bytes(self) -> None:
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, terminator())
        )
        harness.start()
        try:
            results = harness.client.multi_read([])
        finally:
            harness.stop()

        assert results == []
        multi_requests = [
            r for r in harness.server.requests if r[1] == MULTI_READ_OP
        ]
        assert len(multi_requests) == 1
        _, _, body = multi_requests[0]
        assert body == multiheader_struct.pack(-1, 1, -1)

    def test_mixed_success_failure_preserves_order(self) -> None:
        multi_body = (
            getDataResult(b"hello")
            + errorResult(NoNodeError.code)
            + getChildrenResult(["c1", "c2"])
            + terminator()
        )
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, multi_body)
        )
        harness.start()
        try:
            results = harness.client.multi_read(
                [
                    ("data", "/a"),
                    ("data", "/missing"),
                    ("children", "/b"),
                ]
            )
        finally:
            harness.stop()

        assert len(results) == 3
        assert results[0][0] == b"hello"
        assert tuple(results[0][1]) == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        assert isinstance(results[1], NoNodeError)
        assert results[2] == ["c1", "c2"]

        _, _, body = [
            r for r in harness.server.requests if r[1] == MULTI_READ_OP
        ][0]
        assert body == expected_multi_read_body(
            [
                ("/a", GetData.type),
                ("/missing", GetData.type),
                ("/b", GetChildren.type),
            ]
        )

    def test_async_entry_same_results(self) -> None:
        multi_body = (
            getDataResult(b"abc") + getChildrenResult([]) + terminator()
        )
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, multi_body)
        )
        harness.start()
        try:
            async_result = harness.client.multi_read_async(
                [("data", "/a"), ("children", "/empty")]
            )
            results = async_result.get(5)
        finally:
            harness.stop()

        assert results[0][0] == b"abc"
        assert results[1] == []

    def test_chroot_paths_are_prefixed(self) -> None:
        multi_body = (
            getDataResult(b"x") + getChildrenResult(["n"]) + terminator()
        )
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, multi_body),
            chroot="/root/sub",
        )
        harness.start()
        try:
            results = harness.client.multi_read(
                [("data", "/a"), ("children", "/b")]
            )
        finally:
            harness.stop()

        assert results[0][0] == b"x"
        assert results[1] == ["n"]

        _, _, body = [
            r for r in harness.server.requests if r[1] == MULTI_READ_OP
        ][0]
        assert body == expected_multi_read_body(
            [
                ("/root/sub/a", GetData.type),
                ("/root/sub/b", GetChildren.type),
            ]
        )

    def test_unsupported_server_raises_unimplemented(self) -> None:
        # Code -117 is UnimplementedError from a Zookeeper server
        # older than 3.6 that doesn't know opcode 22.
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, -117)
        )
        harness.start()
        try:
            with pytest.raises(UnimplementedApiError):
                harness.client.multi_read([("data", "/a")])
        finally:
            harness.stop()

        opcodes = [r[1] for r in harness.server.requests]
        # Sent exactly once as a single multiRead batch; never split
        # into separate getData/getChildren requests.
        assert opcodes == [MULTI_READ_OP]

    def test_connection_level_error(self) -> None:
        # ConnectionLoss (-4) on the reply header is a connection-level
        # failure with no per-item results at all.
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, ConnectionLoss.code)
        )
        harness.start()
        try:
            with pytest.raises(ConnectionLoss):
                harness.client.multi_read(
                    [("data", "/a"), ("children", "/b")]
                )
        finally:
            harness.stop()

    def test_builder_tracks_chroot_and_order(self) -> None:
        multi_body = (
            getDataResult(b"d") + errorResult(NoNodeError.code) + terminator()
        )
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, multi_body),
            chroot="/root",
        )
        harness.start()
        try:
            batch = harness.client.multi_read()
            batch.get_data("/a")
            batch.get_children("/missing")
            results = batch.commit()
        finally:
            harness.stop()

        assert results[0][0] == b"d"
        assert isinstance(results[1], NoNodeError)

        _, _, body = [
            r for r in harness.server.requests if r[1] == MULTI_READ_OP
        ][0]
        assert body == expected_multi_read_body(
            [
                ("/root/a", GetData.type),
                ("/root/missing", GetChildren.type),
            ]
        )

    def test_double_commit_rejected(self) -> None:
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, terminator())
        )
        harness.start()
        try:
            batch = harness.client.multi_read()
            batch.commit_async().get(5)
            with pytest.raises(ValueError):
                batch.commit()
        finally:
            harness.stop()

    def test_bad_operation_kind_rejected_before_send(self) -> None:
        harness = MultiReadHarness(
            lambda xid, op_type, body: _reply(xid, 0, terminator())
        )
        harness.start()
        try:
            with pytest.raises(ValueError):
                harness.client.multi_read_async(
                    [("stat", "/a")]  # type: ignore[list-item]
                )
            with pytest.raises(TypeError):
                harness.client.multi_read_async(
                    [("/a",)]  # type: ignore[list-item]
                )
            with pytest.raises(TypeError):
                harness.client.multi_read().get_data(  # type: ignore[arg-type]
                    123
                )
        finally:
            harness.stop()
        assert harness.server.requests == []
