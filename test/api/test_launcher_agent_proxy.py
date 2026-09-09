import socket
import threading
import time
import unittest

from fastapi import FastAPI, Request, WebSocket
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from urllib.parse import urlsplit

from fastllm_pytools.launcher_agent_proxy import AgentProxy, create_harness_proxy


class AgentProxyTest(unittest.TestCase):
    def setUp(self):
        import uvicorn
        upstream = FastAPI()
        self.received = []

        @upstream.api_route("/{path:path}", methods=["GET", "POST"])
        async def request(request: Request, path: str):
            self.received.append(dict(request.headers))
            if request.query_params.get("token") == "harness-token":
                response = RedirectResponse("/", status_code=303)
                response.set_cookie("native-harness", "private-cookie", httponly=True)
                return response
            if path == "events":
                async def events():
                    yield b"data: first\n\n"
                    yield b"data: second\n\n"
                return StreamingResponse(events(), media_type="text/event-stream")
            if path == "native":
                return Response('<!doctype html><html><head><script src="/native.js"></script></head>'
                                '<body><main>Native app</main></body></html>', media_type="text/html",
                                headers={"Content-Security-Policy": "script-src 'self'; style-src 'self'", "ETag": '"native"'})
            if path == "native.js":
                return Response('const documentText = "</head>";', media_type="application/javascript")
            return JSONResponse({"path":path, "query":request.url.query, "body":(await request.body()).decode()})

        @upstream.websocket("/terminal")
        async def terminal(websocket: WebSocket):
            self.received.append(dict(websocket.headers))
            await websocket.accept()
            value = await websocket.receive_text()
            await websocket.send_text("echo:" + value)
            await websocket.close()

        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); self.addCleanup(listener.close)
        port = listener.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(upstream, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets":[listener]}, daemon=True)
        self.thread.start(); self.addCleanup(self.stop_server)
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(self.server.started)
        self.proxy = AgentProxy(f"http://127.0.0.1:{port}", "private-password", "127.0.0.1", "http://localhost:8000")
        self.addCleanup(self.proxy.stop)
        self.client = TestClient(self.proxy._app(), base_url=self.proxy.origin)
        self.client.__enter__(); self.addCleanup(self.client.__exit__, None, None, None)

    def stop_server(self):
        self.server.should_exit = True; self.thread.join(timeout=5)

    def test_token_cookie_origin_and_upstream_credentials(self):
        self.assertEqual(self.client.get("/").status_code, 403)
        self.assertEqual(self.client.get("/_ftllm/open?token=wrong").status_code, 403)
        response = self.client.get("/_ftllm/open?token=" + self.proxy.token, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("SameSite=strict", response.headers["set-cookie"])
        self.assertEqual(self.client.post("/session", headers={"origin":"http://malicious.example"}).status_code, 403)
        response = self.client.post("/session?directory=%2Ftmp", content="payload", headers={
            "origin":self.proxy.origin, "X-FTLLM-Launcher-Token":"private-control", "authorization":"Bearer wrong"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"path":"session", "query":"directory=%2Ftmp", "body":"payload"})
        self.assertEqual(self.received[-1]["authorization"], self.proxy.authorization)
        self.assertNotIn("cookie", self.received[-1])
        self.assertNotIn("x-ftllm-launcher-token", self.received[-1])
        self.assertIn("frame-ancestors http://localhost:8000", response.headers["content-security-policy"])

    def test_sse_and_terminal_websocket_forward_without_reauthentication(self):
        self.client.get("/_ftllm/open?token=" + self.proxy.token)
        response = self.client.get("/events")
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.text, "data: first\n\ndata: second\n\n")
        with self.client.websocket_connect(self.proxy.origin.replace("http://", "ws://") + "/terminal",
                                           headers={"origin":self.proxy.origin}) as socket:
            socket.send_text("hello")
            self.assertEqual(socket.receive_text(), "echo:hello")
        self.assertEqual(self.received[-1]["authorization"], self.proxy.authorization)
        self.assertNotIn("cookie", self.received[-1])

    def test_appearance_is_added_only_to_opencode_documents_and_keeps_authentication(self):
        self.assertEqual(self.client.get('/_ftllm/opencode/appearance.js').status_code, 403)
        self.client.get(self.proxy.url)
        document = self.client.get('/native')
        self.assertIn('src="/native.js"', document.text)
        self.assertEqual(document.text.count('src="/_ftllm/opencode/appearance.js"'), 1)
        self.assertIn('href="/_ftllm/opencode/appearance.css"', document.text)
        self.assertIn('data-parents="[&quot;http://localhost:8000&quot;]"', document.text)
        self.assertNotIn(self.proxy.token, document.text)
        self.assertNotIn('etag', document.headers)
        self.assertIn("script-src 'self'; style-src 'self'", document.headers['content-security-policy'])
        self.assertEqual(self.client.get('/native.js').text, 'const documentText = "</head>";')
        count = len(self.received)
        script = self.client.get('/_ftllm/opencode/appearance.js')
        self.assertEqual(script.status_code, 200)
        self.assertIn('javascript', script.headers['content-type'])
        self.assertEqual(self.client.get('/_ftllm/opencode/appearance.css').status_code, 200)
        self.assertEqual(self.client.get('/_ftllm/opencode/other.js').status_code, 404)
        self.assertEqual(len(self.received), count)
        harness = AgentProxy(self.proxy.upstream, None, '127.0.0.1', 'http://localhost:8000',
                             agent='harness', upstream_cookie='private-cookie')
        self.addCleanup(harness.stop)
        with TestClient(harness._app(), base_url=harness.origin) as browser:
            browser.get(harness.url)
            self.assertNotIn('/_ftllm/opencode/', browser.get('/native').text)

    def test_each_launcher_address_has_its_own_authenticated_entry(self):
        alternate_url = self.proxy.url_for("http://127.0.0.1:8000/")
        alternate = urlsplit(alternate_url)
        # A correct token alone cannot register an arbitrary Host header.
        self.assertEqual(self.client.get("/_ftllm/open?token=" + self.proxy.token,
            headers={"host":"unregistered.example:8000"}).status_code, 403)
        self.client.get("/_ftllm/open?token=" + self.proxy.token)
        with TestClient(self.proxy._app(), base_url="http://" + alternate.netloc) as client:
            self.assertEqual(client.get("/").status_code, 403)
            response = client.get(alternate_url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("frame-ancestors http://127.0.0.1:8000", response.headers["content-security-policy"])
            self.assertNotIn("http://localhost:8000", response.headers["content-security-policy"])
            self.assertEqual(client.post("/session", headers={"origin":self.proxy.origin}).status_code, 403)
            with client.websocket_connect("ws://" + alternate.netloc + "/terminal",
                    headers={"origin":"http://" + alternate.netloc}) as websocket:
                websocket.send_text("alternate browser")
                self.assertEqual(websocket.receive_text(), "echo:alternate browser")
        # Registering a second browser does not invalidate the first one.
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("frame-ancestors http://localhost:8000", response.headers["content-security-policy"])

    def test_harness_exchanges_its_token_and_injects_private_cookie_for_http_and_websocket(self):
        proxy = create_harness_proxy(self.proxy.upstream + "/?token=harness-token", "127.0.0.1", "http://localhost:8000")
        self.addCleanup(proxy.stop)
        self.assertNotIn("harness-token", proxy.url)
        with TestClient(proxy._app(), base_url=proxy.origin) as browser:
            self.assertEqual(browser.get("/").status_code, 403)
            response = browser.get(proxy.url, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertNotIn("private-cookie", response.headers["set-cookie"])
            self.assertEqual(browser.get("/events").status_code, 200)
            self.assertEqual(self.received[-1]["cookie"], "native-harness=private-cookie")
            self.assertNotIn("authorization", self.received[-1])
            with browser.websocket_connect(proxy.origin.replace("http://", "ws://") + "/terminal",
                    headers={"origin":proxy.origin}) as websocket:
                websocket.send_text("Harness terminal")
                self.assertEqual(websocket.receive_text(), "echo:Harness terminal")
            self.assertEqual(self.received[-1]["cookie"], "native-harness=private-cookie")
            self.assertNotIn("origin", self.received[-1])
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            create_harness_proxy(self.proxy.upstream + "/?token=wrong", "127.0.0.1", "http://localhost:8000")


if __name__ == "__main__":
    unittest.main()
