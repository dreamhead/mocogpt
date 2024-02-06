from abc import ABC, abstractmethod
from functools import partial
from inspect import Parameter, Signature
from typing import Generic, TypeVar


class Request(ABC):
    def __init__(self, headers, content: dict):
        self._headers = headers
        self._content = content

    @property
    def api_key(self) -> str:
        return self._headers['Authorization'].split('Bearer ')[1]

    @property
    def model(self) -> str:
        return self._content['model']


class Response(ABC):
    def __init__(self, model):
        self._model = model
        self._content = []


T = TypeVar('T', bound=Request)
R = TypeVar('R', bound=Response)


class SessionContext(Generic[T, R]):
    def __init__(self, request: T, response: R):
        self._request = request
        self._response = response

    @property
    def response(self):
        return self._response


class RequestMatcher(Generic[T], ABC):
    @abstractmethod
    def match(self, request: T) -> bool:
        pass


class AllOfMatcher(RequestMatcher):
    def __init__(self, matchers: list[RequestMatcher]):
        self.matchers = matchers

    def match(self, request: Request) -> bool:
        return all(matcher.match(request) for matcher in self.matchers)


class ResponseHandler(Generic[R], ABC):
    @abstractmethod
    def write_response(self, context: SessionContext):
        pass


class AllOfHandler(ResponseHandler):
    def __init__(self, handlers: list[ResponseHandler]):
        self.handlers = handlers

    def write_response(self, context: SessionContext):
        for handler in self.handlers:
            handler.write_response(context)


def make_sig(*names) -> Signature:
    parms = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, default=None)
             for name in names]
    return Signature(parms)


class EndpointMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        clsdict['__request_sig__'] = make_sig(*clsdict.get('_request_params', []))
        clsdict['__response_sig__'] = make_sig(*clsdict.get('_response_params', []))
        return super().__new__(cls, clsname, bases, clsdict)


class SessionSetting:
    def __init__(self, matcher: RequestMatcher, response_sig: Signature, create_handler):
        self._matcher = matcher
        self._handler = None
        self.response_sig = response_sig
        self.create_handler = create_handler

    def response(self, **kwargs):
        self.response_sig.bind(**kwargs)
        self._handler = self.create_handler(**kwargs)


class Endpoint(metaclass=EndpointMeta):
    _matcher_classes = {}
    _handler_classes = {}

    def __init__(self):
        self.sessions = []
        self._create_matchers = partial(self._create_components, self._matcher_classes)
        self._create_handlers = partial(self._create_components, self._handler_classes)

    def request(self, **kwargs) -> SessionSetting:
        self.__request_sig__.bind(**kwargs)
        matcher = self._create_matchers(**kwargs)
        session = SessionSetting(matcher, self.__response_sig__, self._create_handlers)
        self.sessions.append(session)
        return session

    def _create_components(self, component_classes: dict, **kwargs):
        components: list = []

        for param, Component in component_classes.items():
            value = kwargs.get(param)
            if value:
                components.append(Component(value))

        if len(components) == 0:
            raise ValueError('No components specified')

        if len(components) == 1:
            return components[0]

        return AllOfMatcher(components) if isinstance(components[0], RequestMatcher) else AllOfHandler(components)
