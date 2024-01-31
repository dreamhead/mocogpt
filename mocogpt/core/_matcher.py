from mocogpt.core._base import Request, RequestMatcher


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


class ApiKeyMatcher(RequestMatcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def match(self, request: Request) -> bool:
        return request.api_key == self.api_key


class PromptMatcher(RequestMatcher):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def match(self, request: Request) -> bool:
        return request.prompt == self.prompt


class ModelMatcher(RequestMatcher):
    def __init__(self, model: str):
        self.model = model

    def match(self, request: Request) -> bool:
        return request.model == self.model


class TemperatureMatcher(RequestMatcher):
    def __init__(self, temperature: float):
        self._temperature = temperature

    def match(self, request: Request) -> bool:
        return request.temperature == self._temperature
