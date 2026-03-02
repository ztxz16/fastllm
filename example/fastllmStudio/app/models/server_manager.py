import socket
import subprocess
import sys
from typing import Dict, Optional, Tuple

from app.models.model_item import ModelItem


def _find_free_port(start: int = 8100) -> int:
    port = start
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No free port available")


class ServerManager:
    def __init__(self):
        self._servers: Dict[str, Tuple[subprocess.Popen, int, str]] = {}

    def is_running(self, model_id: str) -> bool:
        if model_id not in self._servers:
            return False
        proc, _, _ = self._servers[model_id]
        return proc.poll() is None

    def get_port(self, model_id: str) -> Optional[int]:
        if model_id in self._servers:
            _, port, _ = self._servers[model_id]
            return port
        return None

    def get_model_name(self, model_id: str) -> Optional[str]:
        if model_id in self._servers:
            _, _, name = self._servers[model_id]
            return name
        return None

    def start_server(self, item: ModelItem) -> int:
        if self.is_running(item.model_id):
            return self._servers[item.model_id][1]

        lc = item.launch_config
        port = lc.port if lc.port > 0 else _find_free_port()
        model_name = lc.model_name if lc.model_name else item.name

        cmd = [sys.executable, "-m", "ftllm", "serve", item.path, "--port", str(port)]

        if lc.device:
            cmd.extend(["--device", lc.device])
        if lc.threads > 0:
            cmd.extend(["--threads", str(lc.threads)])
        if lc.dtype and lc.dtype != "auto":
            cmd.extend(["--dtype", lc.dtype])
        if model_name:
            cmd.extend(["--model_name", model_name])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._servers[item.model_id] = (proc, port, model_name)
        return port

    def stop_server(self, model_id: str) -> bool:
        if model_id not in self._servers:
            return False
        proc, _, _ = self._servers[model_id]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        del self._servers[model_id]
        return True

    def get_running_models(self) -> list:
        result = []
        dead_ids = []
        for mid, (proc, port, name) in self._servers.items():
            if proc.poll() is None:
                result.append({"model_id": mid, "port": port, "name": name})
            else:
                dead_ids.append(mid)
        for mid in dead_ids:
            del self._servers[mid]
        return result

    def shutdown_all(self):
        for mid in list(self._servers.keys()):
            self.stop_server(mid)
