"""Exercise idle inference and disconnects without loading GPU models."""
import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

TEST_API_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path = [path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != TEST_API_DIR]
sys.path.insert(0, os.path.abspath(os.path.join(TEST_API_DIR, '..', '..', 'tools')))

from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect
from fastllm_pytools.openai_server.streaming_response import SSEStreamingResponse
from fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion
from fastllm_pytools.openai_server.protocal.openai_protocol import ChatCompletionRequest


async def never_receive():
    await asyncio.Event().wait()


def scope(version='2.4'):
    return {'type': 'http', 'asgi': {'spec_version': version}}


class StreamKeepaliveTest(unittest.IsolatedAsyncioTestCase):
    async def test_idle_heartbeats_preserve_output_and_pending_inference(self):
        release = asyncio.Event()
        state = []
        output = []
        heartbeats = []

        async def source():
            try:
                yield 'data: {"role":"assistant"}\n\n'
                await release.wait()
                yield b'data: {"content":"ok"}\n\n'
                yield 'data: [DONE]\n\n'
                state.append('completed')
            finally:
                state.append('closed')

        async def cleanup():
            self.assertEqual(state, ['completed', 'closed'])
            state.append('cleanup')

        async def send(message):
            body = message.get('body', b'')
            if body.startswith(b':'):
                heartbeats.append(body)
                self.assertEqual(state, [])
                if len(heartbeats) == 3:
                    release.set()
            elif body:
                output.append(body)

        response = SSEStreamingResponse(source(), background=BackgroundTask(cleanup),
                                        heartbeat_interval=0.01)
        await asyncio.wait_for(response(scope(), never_receive, send), 2)
        self.assertGreaterEqual(len(heartbeats), 3)
        self.assertEqual(output, [b'data: {"role":"assistant"}\n\n',
                                  b'data: {"content":"ok"}\n\n',
                                  b'data: [DONE]\n\n'])
        self.assertEqual(state, ['completed', 'closed', 'cleanup'])

    async def test_send_failure_closes_source_before_cleanup(self):
        for fail_on_heartbeat in (False, True):
            with self.subTest(fail_on_heartbeat=fail_on_heartbeat):
                state = []

                async def source():
                    try:
                        yield 'data: started\n\n'
                        await asyncio.Event().wait()
                    finally:
                        await asyncio.sleep(0)
                        state.append('closed')

                async def cleanup():
                    self.assertEqual(state, ['closed'])
                    state.append('cleanup')

                async def send(message):
                    body = message.get('body', b'')
                    if body and (not fail_on_heartbeat or body.startswith(b':')):
                        raise OSError('client disconnected')

                response = SSEStreamingResponse(source(), background=BackgroundTask(cleanup),
                                                heartbeat_interval=0.01)
                with self.assertRaises(ClientDisconnect):
                    await asyncio.wait_for(response(scope(), never_receive, send), 2)
                self.assertEqual(state, ['closed', 'cleanup'])

    async def test_legacy_asgi_disconnect_cleans_up_under_cancel_scope(self):
        disconnected = asyncio.Event()
        state = []

        async def source():
            try:
                await asyncio.Event().wait()
                yield 'unreachable'
            finally:
                await asyncio.sleep(0)
                state.append('closed')

        async def receive():
            await disconnected.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            if message.get('body', b'').startswith(b':'):
                disconnected.set()

        async def cleanup():
            self.assertEqual(state, ['closed'])
            state.append('cleanup')

        response = SSEStreamingResponse(source(), background=BackgroundTask(cleanup),
                                        heartbeat_interval=0.01)
        await asyncio.wait_for(response(scope('2.3'), receive, send), 2)
        self.assertEqual(state, ['closed', 'cleanup'])

    async def test_cancelled_response_joins_pending_inference(self):
        started = asyncio.Event()
        state = []

        async def source():
            try:
                started.set()
                await asyncio.Event().wait()
                yield 'unreachable'
            finally:
                await asyncio.sleep(0)
                state.append('closed')

        async def send(message):
            pass

        async def cleanup():
            self.assertEqual(state, ['closed'])
            state.append('cleanup')

        response = SSEStreamingResponse(source(), background=BackgroundTask(cleanup))
        task = asyncio.create_task(response(scope(), never_receive, send))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(state, ['closed', 'cleanup'])

    async def test_producer_error_is_not_hidden_and_cleanup_runs(self):
        state = []

        async def source():
            try:
                yield 'data: started\n\n'
                raise RuntimeError('inference failed')
            finally:
                state.append('closed')

        async def send(message):
            pass

        async def cleanup():
            self.assertEqual(state, ['closed'])
            state.append('cleanup')

        response = SSEStreamingResponse(source(), background=BackgroundTask(cleanup))
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            await response(scope(), never_receive, send)
        self.assertEqual(state, ['closed', 'cleanup'])

    async def test_cleanup_runs_if_response_headers_cannot_be_sent(self):
        state = []

        async def source():
            self.fail('inference iterator should not have started')
            yield 'unreachable'

        async def send(message):
            raise OSError('connection already closed')

        response = SSEStreamingResponse(source(), background=BackgroundTask(state.append, 'cleanup'))
        with self.assertRaises(ClientDisconnect):
            await response(scope(), never_receive, send)
        self.assertEqual(state, ['cleanup'])

    async def test_cleanup_aborts_only_the_owned_native_handle(self):
        for disconnect_at_terminal in (False, True):
            with self.subTest(disconnect_at_terminal=disconnect_at_terminal):
                aborted = []
                completion = object.__new__(FastLLmCompletion)
                completion.model = SimpleNamespace(
                    abort_handle=aborted.append,
                    get_response_statistics=lambda handle: {
                        'cached_input_tokens': 0, 'missed_input_tokens': 1, 'output_tokens': 1})
                completion.model_name = 'test-model'
                completion.conversation_handles = {'old-request': 7}

                async def model_output():
                    yield 'hello'

                request = ChatCompletionRequest(model='test-model', messages=[], stream=True, max_tokens=16)
                content = completion.chat_completion_stream_generator(
                    request, None, model_output(), 'old-request', 1, False, handle=7,
                    response_statistics={})
                response = SSEStreamingResponse(content, background=BackgroundTask(
                    completion.check_disconnect, None, 'old-request', 7))

                async def send(message):
                    body = message.get('body', b'')
                    if not body:
                        return
                    if not disconnect_at_terminal:
                        raise OSError('client closed early')
                    if b'"usage"' in body:
                        self.assertNotIn('old-request', completion.conversation_handles)
                        completion.conversation_handles['new-request'] = 7
                        raise OSError('client closed after handle was reused')

                with self.assertRaises(ClientDisconnect):
                    await response(scope(), never_receive, send)
                self.assertEqual(aborted, [] if disconnect_at_terminal else [7])
                self.assertEqual(completion.conversation_handles,
                                 {'new-request': 7} if disconnect_at_terminal else {})


if __name__ == '__main__':
    unittest.main()
