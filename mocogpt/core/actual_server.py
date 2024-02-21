import asyncio
import json
import threading
import types

from aiohttp import web

from mocogpt.core._sse import EventSourceResponse
from mocogpt.core.base_server import GptServer
from mocogpt.core.base_typing import Request, SessionContext
from mocogpt.core.chat_completions import Chat, Completions, CompletionsRequest, CompletionsResponse
from mocogpt.core.embeddings import Embeddings, EmbeddingsRequest, EmbeddingsResponse


class Monitor:
    def on_server_start(self, server):
        pass

    def on_server_end(self, server, elapsed):
        pass

    async def on_session_start(self, request):
        pass

    async def on_session_end(self, response):
        pass


class SessionSettingMixin:
    def match(self, request: Request) -> bool:
        return self._matcher.match(request)

    def write_response(self, context: SessionContext):
        self._handler.write_response(context)


def extend_instance(obj, module):
    for key, value in module.__dict__.items():
        if callable(value):
            setattr(obj, key, types.MethodType(value, obj))

    return obj


class ActualGptServer(GptServer):
    def __init__(self, port, monitor: Monitor = Monitor()):
        completions = Completions()
        chat = Chat(completions)
        embeddings = Embeddings()
        super().__init__(chat, embeddings)
        self.runner = None
        self.thread = None
        self.loop = None
        self.sessions = []
        self.port = port
        self.monitor = monitor

    def _before_start(self):
        self.chat.completions.sessions = [extend_instance(session, SessionSettingMixin)
                                          for session in self.chat.completions.sessions]
        self.embeddings.sessions = [extend_instance(session, SessionSettingMixin)
                                    for session in self.embeddings.sessions]

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
        self._before_start()
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
        app.router.add_post('/v1/chat/completions', self.chat_completions_api)
        app.router.add_post('/v1/embeddings', self.embeddings_api)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()

        return self.runner

    async def _stop(self):
        await self.runner.cleanup()

    async def chat_completions_api(self, request: web.Request):
        json_request = await request.json()
        await self.monitor.on_session_start(json_request)
        chat_request = CompletionsRequest(request.headers, json_request)
        response = CompletionsResponse(chat_request.model, chat_request.prompt_tokens())
        context = SessionContext(chat_request, response)

        matched_session = next(
            (session for session in self.chat.completions.sessions if session.match(chat_request)), None)

        if matched_session is None:
            return await self.default_response(request)

        matched_session.write_response(context)
        if chat_request.stream:
            return await self.stream_response(context, request)

        response = web.json_response(context.response.to_dict())
        await self.monitor.on_session_end(response.text)
        return response

    async def embeddings_api(self, request: web.Request):
        json_request = await request.json()
        embeddings_request = EmbeddingsRequest(request.headers, json_request)
        embeddings_response = EmbeddingsResponse(embeddings_request.model)

        context = SessionContext(embeddings_request, embeddings_response)

        try:
            matched_session = next(
                (session for session in self.embeddings.sessions if session.match(embeddings_request)), None)
        except Exception as e:
            print(e)
            return await self.default_response(request)

        if matched_session is None:
            return await self.default_response(request)

        matched_session.write_response(context)
        response = web.json_response(context.response.to_embeddings())
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
