"""Keep inference streams alive and release requests on every exit path."""
import asyncio

import anyio
from starlette.responses import StreamingResponse


async def _with_heartbeats(content, interval):
    iterator = content.__aiter__()
    pending = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                # An SSE comment keeps idle HTTP connections alive without
                # changing model output or creating a protocol event.
                yield ": keep-alive\n\n"
                continue
            completed, pending = pending, None
            try:
                chunk = completed.result()
            except StopAsyncIteration:
                break
            yield chunk
    finally:
        # Waiting for a heartbeat must not cancel inference. A disconnected
        # consumer, however, must cancel and join the one outstanding read.
        with anyio.CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()


class SSEStreamingResponse(StreamingResponse):
    def __init__(self, content, *, background=None, heartbeat_interval=15.0):
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        super().__init__(
            content=_with_heartbeats(content, heartbeat_interval),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self._cleanup = background

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # StreamingResponse does not run its BackgroundTask when sending
            # raises (ASGI 2.4), or when the response task itself is cancelled.
            # Close the iterator before releasing its backend handle, so an
            # outstanding read cannot consume data from a recycled handle.
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    if self._cleanup is not None:
                        await self._cleanup()
