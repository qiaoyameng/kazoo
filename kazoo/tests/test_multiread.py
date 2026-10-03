from __future__ import annotations

import struct
import unittest

import pytest

from kazoo.client import KazooClient
from kazoo.exceptions import (
    ConnectionLoss,
    NoNodeError,
    UnimplementedError,
)
from kazoo.protocol.connection import ConnectionHandler
from kazoo.protocol.serialization import (
    GetChildren,
    GetData,
    MultiRead,
    ReplyHeader,
)
from kazoo.protocol.states import ZnodeStat

_int_struct = struct.Struct("!i")
_multiheader_struct = struct.Struct("!iBi")
_stat_struct = struct.Struct("!qqqqiiiqiiq")

GET_DATA_OP = 4
GET_CHILDREN_OP = 8
ERROR_OP = -1

NO_NODE_ERR = -101
UNIMPLEMENTED_ERR = -6
CONNECTION_LOSS_ERR = -4

STAT_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def _write_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _int_struct.pack(len(encoded)) + encoded


def _write_buffer(value: bytes) -> bytes:
    return _int_struct.pack(len(value)) + value


def _multi_header(type_: int, done: bool, err: int) -> bytes:
    return _multiheader_struct.pack(type_, 1 if done else 0, err)


def _terminator() -> bytes:
    return _multi_header(-1, True, -1)


def _get_data_request(path: str) -> bytes:
    return _write_string(path) + b"\x00"


def _get_children_request(path: str) -> bytes:
    return _write_string(path) + b"\x00"


def _get_data_response(data: bytes) -> bytes:
    return _write_buffer(data) + _stat_struct.pack(*STAT_VALUES)


def _get_children_response(children: list[str]) -> bytes:
    b = _int_struct.pack(len(children))
    for child in children:
        b += _write_string(child)
    return b


def _error_response(err: int) -> bytes:
    return _multi_header(ERROR_OP, False, err) + _int_struct.pack(err)


def _mixed_response_bytes() -> bytes:
    return (
        _multi_header(GET_DATA_OP, False, 0)
        + _get_data_response(b"data-a")
        + _error_response(NO_NODE_ERR)
        + _multi_header(GET_CHILDREN_OP, False, 0)
        + _get_children_response(["c1", "c2"])
        + _terminator()
    )


def _make_ops() -> list:
    return [GetData("/a", None), GetChildren("/b", None)]


class TestMultiReadSerialization(unittest.TestCase):
    def test_serialize_fixed_bytes(self) -> None:
        expected = (
            _multi_header(GET_DATA_OP, False, -1)
            + _get_data_request("/a")
            + _multi_header(GET_CHILDREN_OP, False, -1)
            + _get_children_request("/b")
            + _terminator()
        )
        assert bytes(MultiRead(_make_ops()).serialize()) == expected

    def test_serialize_empty_batch(self) -> None:
        assert bytes(MultiRead([]).serialize()) == _terminator()

    def test_deserialize_fixed_bytes(self) -> None:
        response = (
            _multi_header(GET_DATA_OP, False, 0)
            + _get_data_response(b"data-a")
            + _multi_header(GET_CHILDREN_OP, False, 0)
            + _get_children_response(["c1", "c2"])
            + _terminator()
        )
        results = MultiRead(_make_ops()).deserialize(response, 0)
        assert len(results) == 2
        data, stat = results[0]
        assert data == b"data-a"
        assert stat == ZnodeStat(*STAT_VALUES)
        assert results[1] == ["c1", "c2"]

    def test_deserialize_mixed_results_keep_positions(self) -> None:
        ops = [
            GetData("/a", None),
            GetData("/missing", None),
            GetChildren("/b", None),
        ]
        results = MultiRead(ops).deserialize(_mixed_response_bytes(), 0)
        assert len(results) == 3
        assert results[0][0] == b"data-a"
        assert isinstance(results[1], NoNodeError)
        assert results[2] == ["c1", "c2"]

    def test_deserialize_empty_batch(self) -> None:
        assert MultiRead([]).deserialize(_terminator(), 0) == []

    def test_deserialize_truncated(self) -> None:
        response = (
            _multi_header(GET_DATA_OP, False, 0) + _get_data_response(b"d")
        )
        with pytest.raises(struct.error):
            MultiRead(_make_ops()).deserialize(response, 0)

    def test_deserialize_unknown_op(self) -> None:
        response = _multi_header(99, False, 0) + _terminator()
        with pytest.raises(ValueError):
            MultiRead(_make_ops()).deserialize(response, 0)

    def test_deserialize_too_few_results(self) -> None:
        response = (
            _multi_header(GET_DATA_OP, False, 0)
            + _get_data_response(b"data-a")
            + _terminator()
        )
        with pytest.raises(ValueError):
            MultiRead(_make_ops()).deserialize(response, 0)

    def test_deserialize_too_many_results(self) -> None:
        response = _mixed_response_bytes()
        with pytest.raises(ValueError):
            MultiRead(_make_ops()).deserialize(response, 0)


class FakeConnection:
    """Drives a real ConnectionHandler with fixed response bytes.

    Installs itself as ``client._call`` so requests submitted through
    the public client API are captured, then answered with the fixed
    response bytes through the real response-reading code path.
    """

    def __init__(self, client: KazooClient):
        self.client = client
        self.connection = ConnectionHandler(client, None)
        self.requests: list = []
        self.response_header: ReplyHeader | None = None
        self.response_bytes: bytes = b""
        client._call = self.submit  # type: ignore[method-assign]

    def submit(self, request: object, async_object: object) -> None:
        self.requests.append(request)
        xid = len(self.requests)
        self.client._pending.append((request, async_object, xid))
        header = self.response_header
        if header is None:
            header = ReplyHeader(xid, xid, 0)
        self.connection._read_response(header, self.response_bytes, 0)


def _make_client(chroot: str = "") -> KazooClient:
    return KazooClient(hosts="127.0.0.1:2181" + chroot)


class TestMultiReadClient(unittest.TestCase):
    def test_commit_sends_fixed_bytes_and_returns_results(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_bytes = _mixed_response_bytes()

        mr = client.multi_read()
        mr.get_data("/a")
        mr.get_data("/missing")
        mr.get_children("/b")
        results = mr.commit()

        (request,) = fake.requests
        assert isinstance(request, MultiRead)
        assert request.type == 22
        expected = (
            _multi_header(GET_DATA_OP, False, -1)
            + _get_data_request("/a")
            + _multi_header(GET_DATA_OP, False, -1)
            + _get_data_request("/missing")
            + _multi_header(GET_CHILDREN_OP, False, -1)
            + _get_children_request("/b")
            + _terminator()
        )
        assert bytes(request.serialize()) == expected

        assert len(results) == 3
        assert results[0][0] == b"data-a"
        assert isinstance(results[1], NoNodeError)
        assert results[2] == ["c1", "c2"]

    def test_commit_applies_chroot(self) -> None:
        client = _make_client(chroot="/chroot")
        fake = FakeConnection(client)
        fake.response_bytes = (
            _multi_header(GET_DATA_OP, False, 0)
            + _get_data_response(b"x")
            + _terminator()
        )

        mr = client.multi_read()
        mr.get_data("/a")
        results = mr.commit()

        (request,) = fake.requests
        expected = (
            _multi_header(GET_DATA_OP, False, -1)
            + _get_data_request("/chroot/a")
            + _terminator()
        )
        assert bytes(request.serialize()) == expected
        assert results[0][0] == b"x"

    def test_commit_empty_batch(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_bytes = _terminator()

        results = client.multi_read().commit()

        (request,) = fake.requests
        assert bytes(request.serialize()) == _terminator()
        assert results == []

    def test_commit_async(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_bytes = (
            _multi_header(GET_CHILDREN_OP, False, 0)
            + _get_children_response(["c1"])
            + _terminator()
        )

        mr = client.multi_read()
        mr.get_children("/b")
        async_result = mr.commit_async()
        assert async_result.get() == [["c1"]]

    def test_commit_server_without_multiread_support(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_header = ReplyHeader(1, 1, UNIMPLEMENTED_ERR)

        mr = client.multi_read()
        mr.get_data("/a")
        with pytest.raises(UnimplementedError):
            mr.commit()

    def test_commit_connection_level_error(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_header = ReplyHeader(1, 1, CONNECTION_LOSS_ERR)

        mr = client.multi_read()
        mr.get_data("/a")
        with pytest.raises(ConnectionLoss):
            mr.commit()

    def test_commit_malformed_response_raises(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_bytes = (
            _multi_header(GET_DATA_OP, False, 0) + b"\x00\x00"
        )

        mr = client.multi_read()
        mr.get_data("/a")
        with pytest.raises(struct.error):
            mr.commit()

    def test_double_commit_raises(self) -> None:
        client = _make_client()
        FakeConnection(client)

        mr = client.multi_read()
        mr.get_data("/a")
        mr.commit_async()
        with pytest.raises(ValueError):
            mr.commit_async()
        with pytest.raises(ValueError):
            mr.get_children("/b")

    def test_invalid_path_type(self) -> None:
        client = _make_client()
        mr = client.multi_read()
        with pytest.raises(TypeError):
            mr.get_data(1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            mr.get_children(None)  # type: ignore[arg-type]

    def test_context_manager_commits(self) -> None:
        client = _make_client()
        fake = FakeConnection(client)
        fake.response_bytes = (
            _multi_header(GET_DATA_OP, False, 0)
            + _get_data_response(b"x")
            + _terminator()
        )

        with client.multi_read() as mr:
            mr.get_data("/a")
        assert mr.committed
