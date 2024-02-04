import asyncio
import json
import threading
from abc import ABC

from aiohttp import web

from mocogpt.core._base import (
    Chat,
    Completions,
    CompletionsSessionSetting,
    Embeddings,
    EmbeddingsSessionSetting,
    GptServer,
    Request,
    RequestMatcher,
    RequestSession,
    SessionContext,
    SessionSetting,
)
from mocogpt.core._handler import ContentResponseHandler, EmbeddingsResponseHandler
from mocogpt.core._matcher import (
    AllOfMatcher,
    ApiKeyMatcher,
    EmbeddingsInputMatcher,
    ModelMatcher,
    PromptMatcher,
    TemperatureMatcher,
)
from mocogpt.core._sse import EventSourceResponse


class ActualSessionSetting(SessionSetting):
    def __init__(self, matcher):
        super().__init__(matcher)

    def match(self, request: Request) -> bool:
        return self._matcher.match(request)

    def write_response(self, context: SessionContext):
        self._handler.write_response(context)


class SessionSettingMixin:
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


class ActualCompletionsSessionSetting(CompletionsSessionSetting, SessionSettingMixin):
    def __init__(self, matcher):
        super().__init__(matcher)

    def response(self, content: str | None = None):
        self._handler = ContentResponseHandler(content)


class ActualRequestSession(RequestSession):
    def __init__(self):
        super().__init__()
        self.sessions = []

    def request(self, matcher) -> CompletionsSessionSetting:
        session = ActualCompletionsSessionSetting(matcher)
        self.sessions.append(session)
        return session


class ActualCompletions(Completions):
    def __init__(self):
        self.sessions = []

    def request(self,
                api_key: str | None = None,
                prompt: str | None = None,
                model: str | None = None,
                temperature: float | None = None,
                ) -> CompletionsSessionSetting:
        matcher = self._create_matchers(api_key, model, prompt, temperature)
        session = ActualCompletionsSessionSetting(matcher)
        self.sessions.append(session)
        return session

    def _create_matchers(self, api_key, model, prompt, temperature) -> RequestMatcher:
        matchers = []
        if api_key:
            matchers.append(ApiKeyMatcher(api_key))
        if prompt:
            matchers.append(PromptMatcher(prompt))
        if model:
            matchers.append(ModelMatcher(model))
        if temperature:
            matchers.append(TemperatureMatcher(temperature))
        if len(matchers) == 0:
            raise ValueError("At least one parameter must be provided")
        matcher = matchers[0] if len(matchers) == 1 else AllOfMatcher(matchers)
        return matcher


class ActualEmbeddingsSessionSetting(EmbeddingsSessionSetting, SessionSettingMixin):
    def __init__(self, matcher):
        super().__init__(matcher)

    def response(self, embeddings: list[float]):
        self._handler = EmbeddingsResponseHandler(embeddings)


class ActualEmbeddings(Embeddings):
    def __init__(self):
        self.sessions = []

    def request(self,
                api_key: str | None = None,
                input: str | None = None,
                ) -> EmbeddingsSessionSetting:
        matcher = self._create_matchers(api_key, input)
        session = ActualEmbeddingsSessionSetting(matcher)
        self.sessions.append(session)
        return session

    def _create_matchers(self,
                         api_key: str | None = None,
                         input: str | None = None,
                         ) -> RequestMatcher:
        matchers = []
        if api_key:
            matchers.append(ApiKeyMatcher(api_key))

        if input:
            matchers.append(EmbeddingsInputMatcher(input))

        if len(matchers) == 0:
            raise ValueError("At least one parameter must be provided")

        matcher = matchers[0] if len(matchers) == 1 else AllOfMatcher(matchers)
        return matcher


class ActualGptServer(GptServer):
    def __init__(self, port, monitor: Monitor = Monitor()):
        chat = Chat(ActualCompletions())
        embeddings = ActualEmbeddings()
        super().__init__(chat, embeddings)
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
        app.router.add_post('/v1/chat/completions', self.chat_completions_api)
        app.router.add_post('/v1/embeddings', self.embeddings_api)

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

    async def chat_completions_api(self, request: web.Request):
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

    async def embeddings_api(self, request: web.Request):
        json_request = await request.json()
        embeddings_request = Request(request.headers, json_request)
        context = ActualSessionContext(embeddings_request)

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
