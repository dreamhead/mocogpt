import asyncio
import json
import threading
from abc import ABC

from aiohttp import web

from mocogpt.core._base import Chat, Completions, GptServer, Request, SessionContext, SessionSetting
from mocogpt.core._sse import EventSourceResponse


class ActualSessionSetting(SessionSetting):
    def __init__(self, matcher):
        super().__init__(matcher)

    def match(self, request: Request) -> bool:
        return self._matcher.match(request)

    def write_response(self, context: SessionContext):
        self._handler.write_response(context)


class ActualSessionContext(SessionContext):
    def __init__(self, request: Request):
        super().__init__(request)

    @property
    def stream(self) -> bool:
        return self._request.stream


class Monitor(ABC):
    def on_server_start(self, server):
        pass

    def on_server_end(self, server, elapsed):
        pass

    async def on_session_start(self, request):
        pass

    async def on_session_end(self, response):
        pass


class ActualCompletions(Completions):
    def __init__(self):
        super().__init__()
        self.sessions = []

    def on(self, matcher) -> SessionSetting:
        session = ActualSessionSetting(matcher)
        self.sessions.append(session)
        return session


class ActualGptServer(GptServer):
    def __init__(self, port, monitor: Monitor = Monitor()):
        chat = Chat(ActualCompletions())
        super().__init__(chat)
        self.runner = None
        self.thread = None
        self.loop = None
        self.sessions = []
        self.port = port
        self.monitor = monitor

    def __enter__(self):
        self.thread = threading.Thread(target=self.start_server)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_future = asyncio.run_coroutine_threadsafe(self._stop(), self.loop)
        stop_future.result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

    def start_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.monitor.on_server_start(self)
        self.loop.run_until_complete(self._start())
        self.loop.run_forever()

    def stop_server(self, elapsed):
        asyncio.run_coroutine_threadsafe(self._stop(), self.loop)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.monitor.on_server_end(self, elapsed)

    async def _start(self):
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self.chat_completions)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()

        return self.runner

    async def _stop(self):
        await self.runner.cleanup()

    def completions(self, matcher) -> SessionSetting:
        session = ActualSessionSetting(matcher)
        self.sessions.append(session)
        return session

    def request(self, matcher) -> SessionSetting:
        session = ActualSessionSetting(matcher)
        self.sessions.append(session)
        return session

    async def chat_completions(self, request: web.Request):
        json_request = await request.json()
        await self.monitor.on_session_start(json_request)
        chat_request = Request(request.headers, json_request)
        context = ActualSessionContext(chat_request)

        matched_session = next(
            (session for session in self.chat.completions.sessions if session.match(chat_request)), None)
        if matched_session is None:
            return await self.default_response(request)

        matched_session.write_response(context)
        if context.stream:
            return await self.stream_response(context, request)

        response = web.json_response(context.response.to_dict())
        await self.monitor.on_session_end(response.text)
        return response

    async def stream_response(self, context, request):
        resp = EventSourceResponse()
        await resp.prepare(request)
        async with resp:
            async for data in context.response.sse_content():
                await self.monitor.on_session_end(data)
                await resp.send(json.dumps(data))
        return resp

    async def default_response(self, request):
        response = web.Response(status=400, text="Bad Request")
        await response.prepare(request)
        return response
