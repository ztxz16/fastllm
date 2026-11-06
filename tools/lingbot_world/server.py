#!/usr/bin/env python3
"""LingBot HTTP service. Python standard library only; inference is native CUDA."""
import argparse
import base64
import ctypes as C
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import queue
import re
import secrets
import threading
import time
from urllib.parse import urlsplit

MODEL = 'lingbot-world-2.0-1.3b'
ACTIONS = {'stay', 'forward', 'backward', 'left', 'right', 'up', 'down',
           'look_left', 'look_right', 'look_up', 'look_down'}
CALLBACK = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_int, C.c_int, C.c_void_p)


class Native:
    def __init__(self, library, directory, device, decode_precision="bf16"):
        self.lib = lib = C.CDLL(str(Path(library).resolve()))
        lib.fastllm_lingbot_error.restype = C.c_char_p
        lib.fastllm_lingbot_pipeline_create_ex.argtypes = [C.c_char_p, C.c_int, C.c_int]
        lib.fastllm_lingbot_pipeline_create_ex.restype = C.c_void_p
        lib.fastllm_lingbot_pipeline_destroy.argtypes = [C.c_void_p]
        lib.fastllm_lingbot_world_create_png.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t,
            C.c_int, C.c_int, C.c_uint64, C.POINTER(C.c_float), CALLBACK, C.c_void_p]
        lib.fastllm_lingbot_world_create_png.restype = C.c_void_p
        lib.fastllm_lingbot_world_step.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_char_p, CALLBACK, C.c_void_p]
        lib.fastllm_lingbot_world_step.restype = C.c_void_p
        lib.fastllm_lingbot_world_destroy.argtypes = [C.c_void_p, C.c_void_p]
        lib.fastllm_lingbot_result_frames.argtypes = [C.c_void_p]
        lib.fastllm_lingbot_result_png.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_size_t)]
        lib.fastllm_lingbot_result_png.restype = C.c_void_p
        lib.fastllm_lingbot_result_destroy.argtypes = [C.c_void_p]
        self.handle = lib.fastllm_lingbot_pipeline_create_ex(str(Path(directory).resolve()).encode(), device, int(decode_precision == "bf16"))
        if not self.handle:
            self.fail()

    def fail(self):
        raise RuntimeError(self.lib.fastllm_lingbot_error().decode('utf-8', 'replace'))

    def create(self, options, progress):
        k = (C.c_float*4)(*options['intrinsics']) if options.get('intrinsics') else None
        handle = self.lib.fastllm_lingbot_world_create_png(self.handle, options['prompt'].encode(),
            options['png'], len(options['png']), options['height'], options['width'], options['seed'], k, progress, None)
        if not handle:
            self.fail()
        return handle

    def step(self, world, options, progress):
        prompt = options.get('prompt')
        result = self.lib.fastllm_lingbot_world_step(self.handle, world, options['action'].encode(),
            prompt.encode() if prompt is not None else None, progress, None)
        if not result:
            self.fail()
        try:
            frames = []
            for index in range(self.lib.fastllm_lingbot_result_frames(result)):
                length = C.c_size_t()
                data = self.lib.fastllm_lingbot_result_png(result, index, C.byref(length))
                frames.append(C.string_at(data, length.value))
            return frames
        finally:
            self.lib.fastllm_lingbot_result_destroy(result)

    def delete(self, world):
        if self.lib.fastllm_lingbot_world_destroy(self.handle, world):
            self.fail()


def integer(body, key, default, lo, hi):
    value = body.get(key, default)
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{key} must be an integer in {lo}..{hi}')
    return value


def options(body, create):
    if not isinstance(body, dict):
        raise ValueError('expected a JSON object')
    allowed = {'model', 'prompt', 'action'} | ({'image', 'width', 'height', 'seed', 'intrinsics'} if create else set())
    unknown = set(body)-allowed
    if unknown:
        raise ValueError('unsupported parameters: '+', '.join(sorted(unknown)))
    if body.get('model', MODEL) != MODEL:
        raise ValueError('unknown model')
    result = {'action': body.get('action', 'stay')}
    if not isinstance(result['action'], str) or result['action'] not in ACTIONS:
        raise ValueError('unknown camera action')
    if 'prompt' in body or create:
        prompt = body.get('prompt', '')
        if not isinstance(prompt, str) or '\0' in prompt or len(prompt.encode()) > 16384:
            raise ValueError('prompt must be a string without NUL, at most 16384 UTF-8 bytes')
        result['prompt'] = prompt
    if create:
        result['width'] = integer(body, 'width', 576, 64, 832)
        result['height'] = integer(body, 'height', 320, 64, 832)
        if result['width'] % 16 or result['height'] % 16:
            raise ValueError('width and height must be multiples of 16')
        result['seed'] = integer(body, 'seed', secrets.randbelow(2**31), 0, 2**53-1)
        encoded = body.get('image', '')
        if not isinstance(encoded, str) or not encoded.startswith('data:image/png;base64,'):
            raise ValueError('image must be a PNG data URL')
        try:
            png = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        except (ValueError, base64.binascii.Error) as e:
            raise ValueError('invalid image base64') from e
        if not png.startswith(b'\x89PNG\r\n\x1a\n') or len(png) > 20*1024*1024:
            raise ValueError('invalid PNG image or image exceeds 20 MiB')
        result['png'] = png
        k = body.get('intrinsics')
        if k is not None:
            if not isinstance(k, list) or len(k) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in k) or min(k[:2]) <= 0:
                raise ValueError('intrinsics must be [fx, fy, cx, cy], with positive focal lengths')
            result['intrinsics'] = k
    return result


def chat_options(body):
    allowed = {'model', 'messages', 'world_id', 'action', 'width', 'height', 'seed', 'intrinsics', 'stream'}
    if not isinstance(body, dict) or set(body)-allowed:
        raise ValueError('unsupported chat completion parameters')
    if body.get('stream', False) is not False:
        raise ValueError('streaming chat completions are not supported; use the asynchronous world jobs API')
    messages = body.get('messages')
    if not isinstance(messages, list):
        raise ValueError('messages must be an array')
    users = [m for m in messages if isinstance(m, dict) and m.get('role') == 'user']
    if not users:
        raise ValueError('messages must contain a user message')
    content = users[-1].get('content')
    parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
    if not isinstance(parts, list):
        raise ValueError('user content must be text or an array of text/image_url parts')
    opts = {k: v for k, v in body.items() if k not in {'messages', 'world_id', 'stream'}}
    text = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError('invalid message content part')
        if part.get('type') == 'text' and isinstance(part.get('text'), str):
            text.append(part['text'])
        elif part.get('type') == 'image_url' and isinstance(part.get('image_url'), dict):
            if 'image' in opts:
                raise ValueError('only one starting image is supported')
            opts['image'] = part['image_url'].get('url')
        else:
            raise ValueError('supported message parts are text and image_url')
    if text:
        opts['prompt'] = '\n'.join(text)
    world_id = body.get('world_id')
    if world_id is not None and not isinstance(world_id, str):
        raise ValueError('world_id must be a string')
    return opts, world_id


class Service:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.queue = queue.Queue(8)
        self.worlds, self.jobs = {}, {}
        self.status, self.failure = 'loading', None
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def submit(self, body, world_id=None, delete=False):
        opts = {} if delete else options(body, world_id is None)
        with self.lock:
            if self.status != 'ready':
                raise RuntimeError(self.failure or 'model is loading')
            if world_id is not None:
                if world_id not in self.worlds:
                    raise KeyError('world not found')
                world = self.worlds[world_id]
                if world['pending']:
                    raise BlockingIOError('this world already has a pending action')
            else:
                if len(self.worlds) >= self.args.max_worlds:
                    raise BlockingIOError('world limit reached; delete an unused world first')
                world_id = 'world_'+secrets.token_hex(12)
                world = dict(id=world_id, object='world', width=opts['width'], height=opts['height'],
                    prompt=opts['prompt'], seed=opts['seed'], segments=0, created=int(time.time()), pending=False, handle=None)
            if self.queue.full():
                raise BlockingIOError('generation queue is full')
            job_id = 'job_'+secrets.token_hex(12)
            job = dict(id=job_id, object='world.job', world_id=world_id, status='queued', stage='queued',
                current=0, total=0, created=int(time.time()), error=None, result=None)
            self.worlds[world_id] = world
            world['pending'] = True
            self.jobs[job_id] = job
            self.queue.put_nowait((job, world, opts, delete))
            # Retain bounded job metadata. Frame artifacts remain available on disk.
            done = [key for key, value in self.jobs.items() if value['status'] in ('completed', 'failed')]
            for key in done[:-64]:
                del self.jobs[key]
            return dict(job)

    def run(self):
        try:
            native = Native(self.args.library, self.args.model, self.args.device, self.args.vae_decode_precision)
            with self.lock:
                self.status = 'ready'
            while True:
                job, world, opts, delete = self.queue.get()
                created = world['handle'] is None
                advanced, files = False, []
                @CALLBACK
                def progress(stage, current, total, user):
                    with self.lock:
                        job.update(stage=stage.decode(), current=current, total=total)
                    return 0
                try:
                    with self.lock:
                        job['status'] = 'in_progress'
                    if delete:
                        native.delete(world['handle'])
                        with self.lock:
                            del self.worlds[world['id']]
                            job['result'] = dict(id=world['id'], deleted=True)
                    else:
                        if created:
                            world['handle'] = native.create(opts, progress)
                        frames = native.step(world['handle'], opts, progress)
                        advanced = True
                        frame_urls = []
                        for index, png in enumerate(frames):
                            name = f'{job["id"]}_{index}.png'
                            file = self.output/name
                            files.append(file)
                            file.write_bytes(png)
                            frame_urls.append('/v1/worlds/files/'+name)
                        with self.lock:
                            world['segments'] += 1
                            if 'prompt' in opts:
                                world['prompt'] = opts['prompt']
                            job['result'] = dict(world_id=world['id'], segment=world['segments']-1,
                                width=world['width'], height=world['height'], fps=16, frames=frame_urls, frame_count=len(frames))
                            world['last_result'] = job['result']
                    with self.lock:
                        world['pending'] = False
                        job.update(status='completed', stage='completed')
                except Exception as e:
                    for file in files:
                        try:
                            file.unlink(missing_ok=True)
                        except OSError:
                            pass
                    # Native inference is transactional. Once it succeeds, a
                    # later artifact failure cannot roll that state back: drop
                    # the session instead of reporting stale segment metadata.
                    if created or advanced:
                        if world['handle']:
                            native.delete(world['handle'])
                        with self.lock:
                            self.worlds.pop(world['id'], None)
                    with self.lock:
                        job.update(status='failed', error=str(e), world_deleted=created or advanced)
                finally:
                    with self.lock:
                        world['pending'] = False
                    self.queue.task_done()
        except Exception as e:
            with self.lock:
                self.status, self.failure = 'failed', str(e)

    def get(self, kind, identifier):
        with self.lock:
            item = (self.worlds if kind == 'worlds' else self.jobs).get(identifier)
            if item is None:
                raise KeyError(f'{kind} entry not found')
            return {k: v for k, v in item.items() if k != 'handle'}


def handler(service, demo):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, content_type='application/json'):
            data = json.dumps(value, ensure_ascii=False).encode() if content_type == 'application/json' else value
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def route(self):
            path = urlsplit(self.path).path
            if self.command == 'GET' and path == '/':
                return self.send(200, demo.read_bytes(), 'text/html; charset=utf-8')
            key = service.args.api_key
            if key and self.headers.get('Authorization') != 'Bearer '+key:
                return self.send(401, {'error': {'type': 'authentication_error', 'message': 'invalid API key'}})
            if self.command == 'GET':
                if path == '/health':
                    return self.send(200, dict(status=service.status, error=service.failure))
                if path == '/v1/models':
                    return self.send(200, dict(object='list', data=[dict(id=MODEL, object='model', owned_by='fastllm')]))
                match = re.fullmatch(r'/v1/(worlds|jobs)/((?:world|job)_[0-9a-f]{24})', path)
                if match:
                    return self.send(200, service.get(*match.groups()))
                match = re.fullmatch(r'/v1/worlds/files/(job_[0-9a-f]{24}_[0-9]+\.png)', path)
                if match:
                    file = service.output/match[1]
                    if not file.is_file():
                        raise KeyError('frame not found')
                    return self.send(200, file.read_bytes(), 'image/png')
            if self.command == 'DELETE':
                match = re.fullmatch(r'/v1/worlds/(world_[0-9a-f]{24})', path)
                if match:
                    return self.send(202, service.submit({}, match[1], delete=True))
            if self.command == 'POST':
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 28*1024*1024:
                    raise ValueError('request body must be 1..28 MiB')
                body = json.loads(self.rfile.read(length))
                if path == '/v1/worlds':
                    return self.send(202, service.submit(body))
                if path == '/v1/chat/completions':
                    opts, world_id = chat_options(body)
                    job = service.submit(opts, world_id)
                    deadline = time.monotonic()+900
                    while job['status'] not in ('completed', 'failed'):
                        if time.monotonic() > deadline:
                            return self.send(504, {'error': {'message': 'generation is still running', 'job_id': job['id']}})
                        time.sleep(.25)
                        job = service.get('jobs', job['id'])
                    if job['status'] == 'failed':
                        raise RuntimeError(job['error'])
                    return self.send(200, dict(id='chatcmpl_'+job['id'][4:], object='chat.completion',
                        created=job['created'], model=MODEL, world_id=job['world_id'],
                        choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant',
                            content=json.dumps(job['result'], ensure_ascii=False)))]))
                match = re.fullmatch(r'/v1/worlds/(world_[0-9a-f]{24})/actions', path)
                if match:
                    return self.send(202, service.submit(body, match[1]))
            raise KeyError('endpoint not found')

        def dispatch(self):
            try:
                self.route()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                status = 400 if isinstance(e, (ValueError, TypeError)) else 404 if isinstance(e, KeyError) else 409 if isinstance(e, BlockingIOError) else 500
                self.send(status, {'error': {'type': 'invalid_request_error' if status < 500 else 'server_error', 'message': str(e)}})
        do_GET = do_POST = do_DELETE = dispatch
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18009)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--vae-decode-precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--output', default='lingbot-http-output')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--max-worlds', type=int, default=2)
    parser.add_argument('--demo', type=Path, default=Path(__file__).with_name('demo.html'))
    args = parser.parse_args()
    if not 1 <= args.max_worlds <= 16:
        parser.error('max-worlds must be 1..16')
    service = Service(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(service, args.demo))
    print(f'LingBot demo: http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
