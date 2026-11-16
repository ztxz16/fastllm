"""Output transport concurrency contracts; no GPU or tokenizer is needed."""
import asyncio
import ctypes
import importlib.util
from pathlib import Path
import threading
import unittest

path = Path(__file__).resolve().parents[2]/'tools/fastllm_pytools/async_output.py'
spec = importlib.util.spec_from_file_location('async_output_under_test', path)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


class Function:
    def __init__(self, fn): self.fn = fn
    def __call__(self, *args): return self.fn(*args)


class Backend:
    def __init__(self):
        self.cv = threading.Condition()
        self.rows = {}
        self.version = 0
        self.reader_threads = set()
        self.cancelled = []
        self.get_response_generation_llm_model = Function(self.generation)
        self.abort_response_generation_llm_model = Function(self.abort)
        self.wake_response_output_llm_model = Function(self.wake)
        self.fetch_responses_llm_model = Function(self.fetch)

    def create(self, handle, generation=1):
        with self.cv:
            self.rows[handle] = dict(generation=generation, tokens=[], total=0, status=0)

    def generation(self, model, handle):
        with self.cv: return self.rows.get(handle, {}).get('generation', 0)

    def publish(self, handle, tokens=(), status=0):
        with self.cv:
            row = self.rows[handle]
            row['tokens'].extend(tokens); row['total'] += len(tokens); row['status'] = status
            self.version += 1; self.cv.notify_all()

    def wake(self, model):
        with self.cv:
            self.version += 1; self.cv.notify_all()

    def abort(self, model, handle, generation):
        with self.cv:
            row = self.rows.get(handle)
            if row and row['generation'] == generation:
                self.cancelled.append((handle, generation)); del self.rows[handle]
            self.version += 1; self.cv.notify_all()

    def fetch(self, model, handles, generations, count, limit, tokens, capacity, records, max_records, timeout, version):
        version = ctypes.cast(version, ctypes.POINTER(ctypes.c_uint64))
        with self.cv:
            self.reader_threads.add(threading.get_ident())
            written = used = 0
            for i in range(count):
                row = self.rows.get(handles[i]); status = 0; values = []; stats = (-1,-1,-1)
                if not row or row['generation'] != generations[i]:
                    status = -1
                else:
                    values = row['tokens'][:limit]; del row['tokens'][:limit]
                    stats = (8, 2, row['total'])
                    if row['status'] and not row['tokens']:
                        status = row['status']; del self.rows[handles[i]]
                if values or status:
                    for j,t in enumerate(values): tokens[used+j] = t
                    record = (i,used,len(values),status,*stats)
                    for j,v in enumerate(record): records[written*7+j] = v
                    used += len(values); written += 1
            if not written and version[0] == self.version:
                self.cv.wait(timeout/1000)
            version[0] = self.version
            return written


class OutputTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.lib = Backend()

    async def asyncTearDown(self):
        transport.close_output_reader(self.lib, 1)
        await asyncio.sleep(0)

    def stream(self, h):
        self.lib.create(h)
        return transport.open_output_stream(self.lib, 1, h)

    async def read_all(self, stream):
        result = []
        try:
            while True:
                count,tokens,stats = await asyncio.wait_for(stream.read(),2)
                result.extend(tokens)
                if count < 0: return result,count,stats
        finally: stream.close()

    async def test_many_streams_partial_and_terminal_batches(self):
        streams = [self.stream(h) for h in range(256)]
        for h in range(256): self.lib.publish(h,list(range(130+h%3)),-1)
        results = await asyncio.gather(*(self.read_all(s) for s in streams))
        for h,(tokens,status,stats) in enumerate(results):
            self.assertEqual(tokens,list(range(130+h%3))); self.assertEqual(status,-1)
            self.assertEqual(stats,(8,2,130+h%3))
        self.assertEqual(len(self.lib.reader_threads),1)

    async def test_slow_stream_does_not_block_fast_stream(self):
        slow,fast = self.stream(1),self.stream(2)
        self.lib.publish(1,list(range(200)))
        for i in range(10):
            self.lib.publish(2,[i])
            count,tokens,_ = await asyncio.wait_for(fast.read(),1)
            self.assertEqual((count,tokens),(1,[i]))
        self.assertLessEqual(slow.queue.qsize(),1)
        self.lib.publish(1,status=-1); self.lib.publish(2,status=-1)
        self.assertEqual((await self.read_all(slow))[0],list(range(200)))
        self.assertEqual((await self.read_all(fast))[1],-1)

    async def test_cancel_waiting_read_and_guard_reuse(self):
        stream = self.stream(7)
        task = asyncio.create_task(stream.read())
        while not stream.generation: await asyncio.sleep(.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.lib.create(7,generation=2)
        stream.close()
        await asyncio.sleep(.03)
        self.assertEqual(self.lib.rows[7]['generation'],2)
        self.assertNotIn((7,2),self.lib.cancelled)

    async def test_cancel_one_request_leaves_other_running(self):
        one,two = self.stream(1),self.stream(2)
        while not one.generation or not two.generation: await asyncio.sleep(.001)
        one.close()
        self.lib.publish(2,[42],-1)
        self.assertEqual((await self.read_all(two))[0],[42])
        for _ in range(100):
            if (1,1) in self.lib.cancelled: break
            await asyncio.sleep(.001)
        self.assertIn((1,1),self.lib.cancelled)

    async def test_close_wakes_pending_consumer_and_joins_thread(self):
        stream = self.stream(1)
        task = asyncio.create_task(stream.read())
        while not stream.generation: await asyncio.sleep(.001)
        reader=stream.reader
        transport.close_output_reader(self.lib,1)
        with self.assertRaises(RuntimeError): await asyncio.wait_for(task,1)
        self.assertFalse(reader.thread.is_alive())
        stream.close()

    async def test_error_preserves_tail_then_terminal_statistics(self):
        stream=self.stream(1);self.lib.publish(1,[10,11],-3)
        self.assertEqual(await stream.read(),(2,[10,11],(8,2,2)))
        self.assertEqual(await stream.read(),(-3,[],(8,2,2)))
        stream.close()

    async def test_register_new_request_while_another_is_waiting(self):
        old=self.stream(1)
        while not old.generation: await asyncio.sleep(.001)
        new=self.stream(2);self.lib.publish(2,[9],-1)
        self.assertEqual((await asyncio.wait_for(self.read_all(new),.5))[0],[9])
        old.close()

    async def test_duplicate_subscription_is_rejected(self):
        one=self.stream(1)
        duplicate=transport.open_output_stream(self.lib,1,1)
        with self.assertRaises(RuntimeError): await asyncio.wait_for(duplicate.read(),1)
        duplicate.close()
        self.lib.publish(1,[7],-1)
        self.assertEqual((await self.read_all(one))[0],[7])

    async def test_multiple_event_loops_share_reader(self):
        self.lib.create(1);self.lib.create(2)
        def other_loop():
            async def work():
                s=transport.open_output_stream(self.lib,1,1)
                return await self.read_all(s)
            return asyncio.run(work())
        task=asyncio.create_task(asyncio.to_thread(other_loop))
        s=transport.open_output_stream(self.lib,1,2)
        self.lib.publish(1,[1,2],-1);self.lib.publish(2,[3,4],-1)
        self.assertEqual((await task)[0],[1,2])
        self.assertEqual((await self.read_all(s))[0],[3,4])
        self.assertEqual(len(self.lib.reader_threads),1)

    async def test_old_core_falls_back(self):
        self.assertIsNone(transport.open_output_stream(object(),1,123))


if __name__ == '__main__': unittest.main()
