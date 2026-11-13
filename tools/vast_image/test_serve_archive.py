import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from serve_archive import make_server


class ServeArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = Path(self.temp.name) / "environment.tar"
        self.data = bytes(range(256)) * 8192
        self.archive.write_bytes(self.data)
        self.server = make_server(self.archive, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path="/archive.tar", method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_metadata_and_full_download(self):
        status, _, body = self.request("/metadata")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"name": self.archive.name, "size": len(self.data),
                                           "sha256": hashlib.sha256(self.data).hexdigest()})
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(body, self.data)

    def test_head(self):
        status, headers, body = self.request(method="HEAD", headers={"Range": "bytes=1-2"})
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(self.data))
        self.assertEqual(body, b"")

    def test_ranges_and_resume(self):
        for value, start, end in [("bytes=0-0", 0, 0), ("bytes=100-199", 100, 199),
                                  ("bytes=100-", 100, len(self.data) - 1),
                                  ("bytes=-200", len(self.data) - 200, len(self.data) - 1),
                                  ("bytes=100-99999999", 100, len(self.data) - 1),
                                  ("bytes=-99999999", 0, len(self.data) - 1)]:
            with self.subTest(value=value):
                status, headers, body = self.request(headers={"Range": value})
                self.assertEqual(status, 206)
                self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/{len(self.data)}")
                self.assertEqual(body, self.data[start:end + 1])
                self.assertEqual(int(headers["Content-Length"]), len(body))

    def test_reject_invalid_ranges(self):
        for value in ("bytes=", "bytes=-", "bytes=-0", "bytes=2-1", "bytes=99999999-",
                      "bytes=0-1,4-5", "other=0-1", "bytes=x-y"):
            with self.subTest(value=value):
                status, headers, body = self.request(headers={"Range": value})
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], f"bytes */{len(self.data)}")
                self.assertEqual(body, b"")

    def test_parallel_segments_reconstruct_original(self):
        size = len(self.data) // 16

        def get_segment(index):
            status, _, body = self.request(headers={"Range": f"bytes={index * size}-{(index + 1) * size - 1}"})
            self.assertEqual(status, 206)
            return body

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            output = b"".join(pool.map(get_segment, range(16)))
        self.assertEqual(hashlib.sha256(output).digest(), hashlib.sha256(self.data).digest())

    def test_no_directory_or_other_file_access(self):
        for path in ("/", "/../environment.tar", "/etc/passwd", "/archive.tar/../metadata"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)

    def test_changed_archive_is_rejected(self):
        self.archive.write_bytes(b"x" * len(self.data))
        self.assertEqual(self.request()[0], 409)
        self.assertEqual(self.request("/metadata")[0], 409)

    def test_only_listens_on_loopback(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_reject_empty_file_and_directory(self):
        empty = Path(self.temp.name) / "empty.tar"
        empty.touch()
        for path in (empty, Path(self.temp.name)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                make_server(path, 0)


if __name__ == "__main__":
    unittest.main()
