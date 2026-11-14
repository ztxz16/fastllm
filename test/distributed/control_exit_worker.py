#!/usr/bin/env python3
"""仅用于控制面故障测试的 worker；不加载 CUDA，不替代正式推理 worker。"""

import os
from pathlib import Path
import socket
import struct
import sys
import time


MAGIC = 0x46545031
VERSION = 5
HEADER = struct.Struct("<IIIIQQ")
HELLO, INITIALIZE, READY, PING, PONG, SHUTDOWN, BYE, ERROR = 1, 4, 5, 6, 7, 8, 9, 10
ERROR_TEXT = b"injected worker failure before exit"


def frame(kind, session, sequence, payload=b""):
    return HEADER.pack(MAGIC, VERSION, kind, len(payload), session, sequence) + payload


def receive(connection):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            data = connection.recv(size - len(result))
            if not data:
                raise EOFError("coordinator disconnected")
            result.extend(data)
        return bytes(result)

    magic, version, kind, length, session, sequence = HEADER.unpack(exact(HEADER.size))
    if magic != MAGIC or version != VERSION or length > 1024 * 1024:
        raise ValueError("invalid test command")
    return kind, session, sequence, exact(length)


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("test worker barrier timed out")
        time.sleep(0.005)


def process_exited(pid):
    try:
        # 子进程由 C++ 协调器回收；僵尸也表示已经退出。
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].startswith("Z")
    except FileNotFoundError:
        return True


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    if args["--control-only"] != "1" or args["--world-size"] != "2":
        raise ValueError("exit fixture requires control-only TP=2")
    rank = int(args["--rank"])
    session = int(args["--session"])
    mode = os.environ["FASTLLM_TEST_EXIT_MODE"]
    root = Path(os.environ["FASTLLM_TEST_EXIT_DIR"])
    marker = root / f"{session}.waiting"
    own_pid = root / f"{session}.{rank}.pid"
    own_pid.write_text(str(os.getpid()))
    connection = socket.socket(fileno=int(args["--control-fd"]))
    connection.setblocking(True)

    if mode == "startup_peer_exit":
        if rank == 0:
            marker.touch()
            # 等待父进程关闭连接；不发送 Hello，令其阻塞在 rank 0。
            receive(connection)
        else:
            wait_for(marker.exists)
        return

    hello = struct.pack("<IIIIII", rank, int(args["--device"]), os.getpid(), 0, 0, 0)
    connection.sendall(frame(HELLO, session, 0, hello + bytes(16) + b"control exit fixture"))
    while True:
        kind, message_session, sequence, payload = receive(connection)
        if message_session != session or payload:
            raise ValueError("unexpected test command")
        if kind == INITIALIZE:
            connection.sendall(frame(READY, session, sequence))
        elif kind == SHUTDOWN:
            if mode == "shutdown_order" and rank == 0:
                peer = root / f"{session}.1.pid"
                wait_for(peer.exists)
                wait_for(lambda: process_exited(int(peer.read_text())))
            connection.sendall(frame(BYE, session, sequence))
            return
        elif kind == PING:
            if mode == "shutdown_order":
                connection.sendall(frame(PONG, session, sequence))
            elif mode == "current_error_exit" and rank == 0:
                connection.sendall(frame(ERROR, session, sequence, ERROR_TEXT))
                return
            elif mode == "reply_then_exit":
                if rank == 1:
                    peer = root / f"{session}.0.pid"
                    wait_for(peer.exists)
                    wait_for(lambda: process_exited(int(peer.read_text())))
                connection.sendall(frame(PONG, session, sequence))
                if rank == 0:
                    return
            elif rank == 0:
                if mode == "partial_header_peer_exit":
                    connection.sendall(frame(PONG, session, sequence)[:8])
                elif mode == "partial_payload_peer_exit":
                    connection.sendall(frame(PONG, session, sequence, b"12345678")[:-4])
                marker.touch()
                # 不回复本次 Ping，继续读命令；失败清理时 EOF 会结束本进程。
            elif mode in ("stall", "current_error_exit"):
                connection.sendall(frame(PONG, session, sequence))
            else:
                wait_for(marker.exists)
                if mode in ("buffered_error_exit", "partial_error_exit", "stale_error_exit"):
                    reply = frame(ERROR, session, sequence + (mode == "stale_error_exit"), ERROR_TEXT)
                    connection.sendall(reply[:-4] if mode == "partial_error_exit" else reply)
                return
        else:
            raise ValueError(f"unexpected test command kind={kind}")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
