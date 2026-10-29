"""Transport tests use an explicit fake backend, never a fake inference result."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zlib

from fastapi.testclient import TestClient

path = Path(__file__).resolve().parents[2] / "tools/fastllm_pytools/nomas_server.py"
spec = importlib.util.spec_from_file_location("nomas_http", path)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


class FakeBackend:
    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.closed = False
        self.fail = False
        self.committing = threading.Event()
        self.finish = threading.Event()
        self.finish.set()
        self.dropped = []
        self.requests = []

    def generate(self, request, progress):
        self.requests.append(request)
        self.started.set()
        while not self.release.wait(.01):
            if progress(2, 0, 8):
                raise RuntimeError("generation cancelled")
        if self.fail:
            raise RuntimeError("test backend failure")
        if progress(4, 1, 1):
            raise RuntimeError("generation cancelled")
        self.committing.set()
        self.finish.wait(5)
        return bytes([20, 40, 60]) * request.width * request.height * request.num_frames

    def drop_world(self, world_id):
        self.dropped.append(world_id)

    def close(self):
        self.closed = True


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.backend = FakeBackend()
        self.client = TestClient(server.create_app(self.backend, self.directory.name))
        self.client.__enter__()
        self.request = dict(messages=[dict(role="user", content="A red ball")], width=64, height=64, num_frames=4)

    def tearDown(self):
        self.backend.release.set()
        self.backend.finish.set()
        self.client.__exit__(None, None, None)
        self.assertTrue(self.backend.closed)
        self.directory.cleanup()

    def submit(self):
        response = self.client.post("/v1/videos", json=self.request)
        self.assertEqual(response.status_code, 202)
        return response.json()["id"]

    def terminal(self, job):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            data = self.client.get("/v1/videos/" + job).json()
            if data["status"] in ("completed", "failed", "cancelled"):
                return data
            time.sleep(.01)
        self.fail("job did not terminate")

    def test_real_png_transport_and_validation(self):
        self.backend.release.set()
        job = self.submit()
        result = self.terminal(job)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["output"]["frames"]), 4)
        image = self.client.get(result["output"]["frames"][0])
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertTrue(image.content.startswith(b"\x89PNG\r\n\x1a\n"))
        pos, compressed = 8, b""
        while pos < len(image.content):
            length = struct.unpack(">I", image.content[pos:pos+4])[0]
            kind, payload = image.content[pos+4:pos+8], image.content[pos+8:pos+8+length]
            if kind == b"IDAT":
                compressed += payload
            pos += 12 + length
        self.assertEqual(zlib.decompress(compressed), (b"\0" + bytes([20, 40, 60])*64)*64)
        self.assertEqual(self.client.get(f"/v1/videos/{job}/frames/-1").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/videos/{job}/frames/4").status_code, 404)
        for update in [dict(width=65), dict(prompt="both prompt and messages"), dict(camera=[]), dict(model="other")]:
            self.assertEqual(self.client.post("/v1/videos", json={**self.request, **update}).status_code, 422)
        self.assertEqual(self.client.get("/v1/videos/unknown").status_code, 404)
        self.assertIn("/v1/videos", self.client.get("/openapi.json").json()["paths"])

    def test_cancel_and_failure_do_not_publish_frames(self):
        job = self.submit()
        self.assertTrue(self.backend.started.wait(2))
        self.client.post(f"/v1/videos/{job}/cancel")
        self.client.post(f"/v1/videos/{job}/cancel")
        self.assertEqual(self.terminal(job)["status"], "cancelled")
        self.assertEqual(self.client.get(f"/v1/videos/{job}/frames/0").status_code, 409)
        self.backend.fail = True
        self.backend.release.set()
        other = self.submit()
        result = self.terminal(other)
        self.assertEqual(result["status"], "failed")
        self.assertIn("test backend failure", result["error"]["message"])
        self.assertFalse((Path(self.directory.name) / other).exists())

    def test_queue_is_bounded(self):
        self.submit()
        self.assertTrue(self.backend.started.wait(2))
        for _ in range(4):
            self.submit()
        self.assertEqual(self.client.post("/v1/videos", json=self.request).status_code, 429)

    def test_incremental_frames_sse_and_timing(self):
        encoding = threading.Event()
        finish = threading.Event()
        original = server.png_rgb
        calls = 0
        def encode(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                encoding.set()
                if not finish.wait(5):
                    raise RuntimeError('test did not release PNG encoding')
            return original(*args)
        self.backend.release.set()
        with patch.object(server, 'png_rgb', side_effect=encode):
            job = self.submit()
            try:
                self.assertTrue(encoding.wait(2))
                snapshot = self.client.get('/v1/videos/'+job).json()
                self.assertEqual(snapshot['frames_ready'], 1)
                self.assertEqual(snapshot['status'], 'in_progress')
                self.assertEqual(self.client.get(f'/v1/videos/{job}/frames/0').status_code, 200)
                self.assertEqual(self.client.get(f'/v1/videos/{job}/frames/1').status_code, 409)
            finally:
                finish.set()
            self.assertEqual(self.terminal(job)['status'], 'completed')
        response = self.client.get(f'/v1/videos/{job}/events')
        self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))
        result = json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith('data: ')))
        self.assertEqual(result['frames_ready'], 4)
        self.assertEqual(result['status'], 'completed')
        self.assertGreater(result['generation_fps'], 0)
        self.assertGreaterEqual(result['total_seconds'], result['generation_seconds']-.001)
        self.assertEqual(self.client.get('/v1/videos/unknown/events').status_code, 404)

    def test_world_resolution_and_cache_budget(self):
        self.backend.release.set()
        request = dict(prompt='A room', width=512, height=288)
        response = self.client.post('/v1/worlds', json=request)
        self.assertEqual(response.status_code, 202)
        self.terminal(response.json()['current_job'])
        self.assertEqual(self.client.post('/v1/worlds', json=dict(prompt='A room',width=64,height=64)).status_code, 429)
        self.assertEqual(self.client.post('/v1/worlds', json={**request, 'height':512}).status_code, 422)
        self.assertEqual(self.client.delete('/v1/worlds/'+response.json()['id']).status_code, 200)

    def world(self):
        response = self.client.post('/v1/worlds', json={**self.request, 'num_frames':20})
        self.assertEqual(response.status_code,202)
        return response.json()

    def test_world_continuation_busy_reset_and_latest_retention(self):
        world = self.world();wid=world['id']
        self.assertTrue(self.backend.started.wait(2))
        self.assertEqual(self.client.post(f'/v1/worlds/{wid}/actions',json={'action':'forward'}).status_code,409)
        self.assertEqual(self.client.delete(f'/v1/worlds/{wid}').status_code,409)
        self.backend.release.set()
        first=self.terminal(world['current_job'])
        self.assertEqual((first['world_id'],first['segment_index'],first['start_frame']),(wid,1,0))
        action=self.client.post(f'/v1/worlds/{wid}/actions',json={'action':'forward'})
        self.assertEqual(action.status_code,202)
        second=self.terminal(action.json()['id'])
        self.assertEqual((second['action'],second['segment_index'],second['start_frame']),('forward',2,20))
        state=self.client.get(f'/v1/worlds/{wid}').json()
        self.assertEqual((state['status'],state['segments'],state['generated_frames']),('ready',2,40))
        self.assertNotIn('request',state)
        # Unrelated video jobs must not evict the active world's latest view.
        self.client.app.state.jobs.max_results=3
        for _ in range(5):self.terminal(self.submit())
        self.assertEqual(self.client.get('/v1/videos/'+state['latest_video']).status_code,200)
        self.assertEqual(self.client.delete(f'/v1/worlds/{wid}').status_code,200)
        self.assertIn(wid,self.backend.dropped)
        self.assertEqual(self.client.get(f'/v1/worlds/{wid}').status_code,404)

    def test_world_cancel_does_not_advance(self):
        world=self.world();wid=world['id']
        self.assertTrue(self.backend.started.wait(2))
        self.client.post('/v1/videos/'+world['current_job']+'/cancel')
        self.assertEqual(self.terminal(world['current_job'])['status'],'cancelled')
        state=self.client.get(f'/v1/worlds/{wid}').json()
        self.assertEqual((state['status'],state['segments']),('ready',0))
        self.backend.release.set()
        retry=self.client.post(f'/v1/worlds/{wid}/actions',json={'action':'stay'}).json()
        self.assertEqual(self.terminal(retry['id'])['segment_index'],1)

    def test_world_commit_barrier_and_save_failure(self):
        self.backend.release.set();self.backend.finish.clear()
        world=self.world();wid=world['id']
        self.assertTrue(self.backend.committing.wait(2))
        self.assertEqual(self.client.post('/v1/videos/'+world['current_job']+'/cancel').status_code,409)
        self.backend.finish.set()
        self.assertEqual(self.terminal(world['current_job'])['status'],'completed')
        with patch.object(server,'png_rgb',side_effect=OSError('test disk full')):
            job=self.client.post(f'/v1/worlds/{wid}/actions',json={'action':'forward'}).json()
            self.assertEqual(self.terminal(job['id'])['status'],'failed')
        self.assertEqual(self.client.get(f'/v1/worlds/{wid}').json()['status'],'failed')
        self.assertEqual(self.client.post(f'/v1/worlds/{wid}/actions',json={'action':'forward'}).status_code,409)
        self.assertEqual(self.client.delete(f'/v1/worlds/{wid}').status_code,200)

    def test_world_capacity_and_action_validation(self):
        world=self.world();self.world()
        self.assertEqual(self.client.post('/v1/worlds',json={**self.request,'num_frames':20}).status_code,429)
        self.assertEqual(self.client.post('/v1/worlds',json=self.request).status_code,422)
        self.assertEqual(self.client.post(f"/v1/worlds/{world['id']}/actions",json={'action':'teleport'}).status_code,422)

    def test_text_only_continuation_retains_history_and_action(self):
        self.backend.release.set()
        world=self.world();wid=world['id']
        self.terminal(world['current_job'])
        result=self.client.post(f'/v1/worlds/{wid}/actions',json={'action':'follow_prompt'})
        self.assertEqual(result.status_code,202)
        job=self.terminal(result.json()['id'])
        self.assertEqual((job['world_id'],job['segment_index'],job['action']),(wid,2,'follow_prompt'))
        self.assertEqual(job['text_condition'],'scene_description')
        self.assertEqual(self.backend.requests[-1].action,'follow_prompt')
        self.assertEqual(server.ACTIONS.index('follow_prompt'),9)

    def test_edit_once_then_explore_and_explicitly_follow_text(self):
        self.backend.release.set();world=self.world();url='/v1/worlds/'+world['id']
        self.assertEqual(self.terminal(world['current_job'])['text_condition'],'scene_description')
        self.client.patch(url,json={'prompt':'A cat appears on the table'})
        def act(action):
            response=self.client.post(url+'/actions',json={'action':action})
            self.assertEqual(response.status_code,202)
            return self.terminal(response.json()['id'])
        self.assertEqual(act('follow_prompt')['text_condition'],'scene_description')
        for action in server.ACTIONS[:-1]:
            self.assertEqual(act(action)['text_condition'],'camera_continuation')
        self.assertEqual(act('follow_prompt')['text_condition'],'scene_description')
        self.assertEqual(act('left')['text_condition'],'camera_continuation')
        # A new description can also be applied together with a camera action.
        self.client.patch(url,json={'prompt':'The cat jumps off the table'})
        self.assertEqual(act('right')['text_condition'],'scene_description')
        self.assertEqual(act('right')['text_condition'],'camera_continuation')
        state=self.client.get(url).json()
        self.assertEqual(state['applied_prompt'],'The cat jumps off the table')
        self.assertEqual(state['last_text_condition'],'camera_continuation')
        self.assertFalse(self.backend.dropped)

    def test_live_prompt_retains_world_and_running_request(self):
        world=self.world();wid=world['id'];url='/v1/worlds/'+wid
        self.assertTrue(self.backend.started.wait(2))
        edited=self.client.patch(url,json={'messages':[{'role':'user','content':'A blue ball'}]})
        self.assertEqual(edited.status_code,200)
        self.assertEqual((edited.json()['id'],edited.json()['segments'],edited.json()['prompt_revision']),(wid,0,2))
        self.assertEqual(self.backend.requests[0].caption(),'A red ball')
        self.backend.release.set();first=self.terminal(world['current_job'])
        self.assertEqual(first['prompt_revision'],1)
        state=self.client.get(url).json()
        self.assertEqual((state['prompt'],state['applied_prompt']),('A blue ball','A red ball'))
        action=self.client.post(url+'/actions',json={'action':'look_left'}).json()
        second=self.terminal(action['id'])
        self.assertEqual((second['segment_index'],second['start_frame'],second['prompt_revision']),(2,20,2))
        self.assertEqual(self.backend.requests[-1].caption(),'A blue ball')
        state=self.client.get(url).json()
        self.assertEqual((state['id'],state['segments'],state['applied_prompt_revision']),(wid,2,2))
        self.assertEqual(self.client.patch(url,json={'prompt':'A blue ball'}).json()['prompt_revision'],2)
        self.assertFalse(self.backend.dropped)

    def test_live_prompt_cancel_and_superseding_edit(self):
        self.backend.release.set();world=self.world();wid=world['id'];url='/v1/worlds/'+wid
        self.terminal(world['current_job'])
        self.backend.release.clear();self.backend.started.clear()
        self.client.patch(url,json={'prompt':'A blue ball'})
        job=self.client.post(url+'/actions',json={'action':'stay'}).json()
        self.assertEqual(job['text_condition'],'scene_description')
        self.assertTrue(self.backend.started.wait(2))
        self.client.patch(url,json={'prompt':'A green ball'})
        self.assertEqual(self.backend.requests[-1].caption(),'A blue ball')
        self.client.post('/v1/videos/'+job['id']+'/cancel')
        self.assertEqual(self.terminal(job['id'])['status'],'cancelled')
        state=self.client.get(url).json()
        self.assertEqual((state['segments'],state['applied_prompt_revision'],state['prompt_revision']),(1,1,3))
        self.backend.release.set()
        retry=self.client.post(url+'/actions',json={'action':'stay'}).json()
        self.assertEqual(retry['text_condition'],'scene_description')
        self.assertEqual(self.terminal(retry['id'])['prompt_revision'],3)
        state=self.client.get(url).json()
        self.assertEqual((state['segments'],state['applied_prompt']),(2,'A green ball'))

    def test_live_prompt_validation_and_failed_world(self):
        self.backend.release.set();world=self.world();url='/v1/worlds/'+world['id'];self.terminal(world['current_job'])
        for body in ({},{'prompt':' '},{'prompt':'x','messages':[{'role':'user','content':'y'}]},
                     {'prompt':'x','width':128},{'prompt':'你'*6000}):
            self.assertEqual(self.client.patch(url,json=body).status_code,422)
        self.assertEqual(self.client.patch('/v1/worlds/missing',json={'prompt':'x'}).status_code,404)
        with patch.object(server,'png_rgb',side_effect=OSError('disk full')):
            job=self.client.post(url+'/actions',json={'action':'stay'}).json();self.terminal(job['id'])
        self.assertEqual(self.client.patch(url,json={'prompt':'x'}).status_code,409)


if __name__ == "__main__":
    unittest.main()
