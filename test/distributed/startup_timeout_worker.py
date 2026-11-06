#!/usr/bin/env python3
"""故障夹具：制造超时或错误握手，不改变正式 worker 的行为。"""

import os
import socket
import struct
import sys
import time

fault = os.environ.get("FASTLLM_TEST_STARTUP_FAULT", "timeout")
if fault == "timeout":
    time.sleep(120)
else:
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    connection = socket.socket(fileno=int(args["--control-fd"]))
    connection.setblocking(True)
    session = int(args["--session"])
    version = 2 if fault == "protocol_version" else 4
    if fault == "protocol_session":
        session ^= 1
    payload = struct.pack("<IIIIII", int(args["--rank"]), int(args["--device"]), os.getpid(), 0, 0, 0)
    payload += bytes(16) + b"test-only fault worker"
    header = struct.pack("<IIIIQQ", 0x46545031, version, 1, len(payload), session, 0)
    connection.sendall(header + payload)
    time.sleep(120)
