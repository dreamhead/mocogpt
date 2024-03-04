import re
from abc import ABC, abstractmethod
from enum import Enum, unique
from functools import partial
from inspect import Parameter, Signature
from typing import Generic, TypeVar


@unique
class UnaryOperatorType(Enum):
    EQUALS = 0
    CONTAINS = 1
    STARTSWITH = 2
    ENDSWITH = 3
    REGEX = 4


class UnaryOperator:
    operators = {
        UnaryOperatorType.EQUALS: lambda value, arg: value == arg,
        UnaryOperatorType.CONTAINS: lambda value, arg: arg in value,
        UnaryOperatorType.STARTSWITH: lambda value, arg: value.startswith(arg),
        UnaryOperatorType.ENDSWITH: lambda value, arg: value.endswith(arg),
        UnaryOperatorType.REGEX: lambda value, arg: re.match(arg, value) is not None
    }

    def __init__(self, _type: UnaryOperatorType, arg):
        self.type = _type
        self.arg = arg

    def match(self, value):
        return UnaryOperator.operators[self.type](value, self.arg)


@unique
class VarargOperatorType(Enum):
    ANY_OF = 1
    NONE_OF = 2


class VarargOperator:
    def __init__(self, type: VarargOperatorType, *args):
        self.type = type
        self.args = args


def any_of(*args):
    return VarargOperator(VarargOperatorType.ANY_OF, *args)


def none_of(*args):
    return VarargOperator(VarargOperatorType.NONE_OF, *args)


def eq(arg):
    return UnaryOperator(UnaryOperatorType.EQUALS, arg)


def contains(arg):
    return UnaryOperator(UnaryOperatorType.CONTAINS, arg)


def startswith(arg):
    return UnaryOperator(UnaryOperatorType.STARTSWITH, arg)


def endswith(arg):
    return UnaryOperator(UnaryOperatorType.ENDSWITH, arg)


def regex(arg):
    return UnaryOperator(UnaryOperatorType.REGEX, arg)


class RequestMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        _content_fields: list[str] = clsdict.get('_content_fields', [])
        for field in _content_fields:
            clsdict[field] = property(
                lambda self, _field=field: self._content[_field] if _field in self._content else None)

        return super().__new__(cls, clsname, bases, clsdict)


class Request(metaclass=RequestMeta):
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


class AnyOfMatcher(RequestMatcher):
    def __init__(self, matchers: list[RequestMatcher]):
        self.matchers = matchers

    def match(self, request: Request) -> bool:
        return any(matcher.match(request) for matcher in self.matchers)


class NoneOfMatcher(RequestMatcher):
    def __init__(self, matchers: list[RequestMatcher]):
        self.matchers = matchers

    def match(self, request: Request) -> bool:
        return not any(matcher.match(request) for matcher in self.matchers)


class RequestExtractor(Generic[T], ABC):
    @abstractmethod
    def extract(self, request: T) -> str:
        pass


class UnaryOperatorMatcher(RequestMatcher):
    def __init__(self, extractor: RequestExtractor, operator: UnaryOperator):
        self.extractor = extractor
        self.operator = operator

    def match(self, request: Request) -> bool:
        return self.operator.match(self.extractor.extract(request))


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


class SessionSetting:
    def __init__(self, matcher: RequestMatcher,
                 create_matcher,
                 create_handler):
        self._matcher = matcher
        self._handler = None
        self._create_handler = create_handler
        self._create_matcher = create_matcher

    def or_request(self, **kwargs):
        self._matcher = AnyOfMatcher([self._matcher, self._create_matcher(**kwargs)])
        return self

    def response(self, **kwargs):
        self._handler = self._create_handler(**kwargs)


class EndpointMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        request_params: dict = clsdict.get('_request_params', {})
        clsdict['__request_sig__'] = make_sig(*request_params.keys())
        response_params: dict = clsdict.get('_response_params', [])
        clsdict['__response_sig__'] = make_sig(*response_params.keys())
        return super().__new__(cls, clsname, bases, clsdict)


class Endpoint(metaclass=EndpointMeta):
    _request_params = {}
    _response_params = {}

    def __init__(self):
        self.sessions = []
        self._create_matchers = partial(self._create_components, self.__request_sig__, self._request_params)
        self._create_handlers = partial(self._create_components, self.__response_sig__, self._response_params)

    def request(self, **kwargs) -> SessionSetting:
        matcher = self._create_matchers(**kwargs)
        session = SessionSetting(matcher, self._create_matchers, self._create_handlers)
        self.sessions.append(session)
        return session

    def _create_component(self, Component: type, value):
        if isinstance(value, VarargOperator):
            if value.type == VarargOperatorType.ANY_OF:
                return AnyOfMatcher([self._create_component(Component, value) for value in value.args])

            if value.type == VarargOperatorType.NONE_OF:
                return NoneOfMatcher([self._create_component(Component, value) for value in value.args])

            raise ValueError(f'Unknown operator type: {value.type}')

        if isinstance(value, UnaryOperator):
            return UnaryOperatorMatcher(Component(), value)

        return self._do_create_component(Component, value)

    def _do_create_component(self, Component, value):
        if issubclass(Component, RequestExtractor):
            return UnaryOperatorMatcher(Component(), eq(value))

        return Component(value)

    def _create_components(self, sig: Signature, component_classes: dict, **kwargs):
        sig.bind(**kwargs)
        components = [self._create_component(Component, value)
                      for param, Component in component_classes.items() if (value := kwargs.get(param))]

        if len(components) == 0:
            raise ValueError('No components specified')

        if len(components) == 1:
            return components[0]

        return AllOfMatcher(components) if isinstance(components[0], RequestMatcher) else AllOfHandler(components)


def to_class_name(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


extractors = {}


def extractor_class(name):
    global extractors
    if name in extractors:
        return extractors[name]

    clazz = type(to_class_name(name) + 'Extractor', (RequestExtractor,), {
        'extract': lambda self, request: getattr(request, name.lower())
    })

    extractors[name] = clazz
    return clazz
