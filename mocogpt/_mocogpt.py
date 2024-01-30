from mocogpt.core._base import GptServer, RequestMatcher, ResponseHandler
from mocogpt.core._handler import ContentResponseHandler
from mocogpt.core._matcher import AllOfMatcher, AnyOfMatcher, ModelMatcher, PromptMatcher, TemperatureMatcher
from mocogpt.core._server import ActualGptServer


def gpt_server(port) -> GptServer:
    return ActualGptServer(port)


def content(text: str) -> ResponseHandler:
    return ContentResponseHandler(text)


def all_of(*matchers: RequestMatcher) -> RequestMatcher:
    return AllOfMatcher(matchers)


def any_of(*matchers: RequestMatcher) -> RequestMatcher:
    return AnyOfMatcher(matchers)


def prompt(text: str) -> RequestMatcher:
    return PromptMatcher(text)


def model(text: str) -> RequestMatcher:
    return ModelMatcher(text)


def temperature(temp: float) -> RequestMatcher:
    return TemperatureMatcher(temp)
