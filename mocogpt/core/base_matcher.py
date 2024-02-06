from mocogpt.core.base_typing import Request, RequestMatcher


class ApiKeyMatcher(RequestMatcher[Request]):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def match(self, request: Request) -> bool:
        return request.api_key == self.api_key


class ModelMatcher(RequestMatcher[Request]):
    def __init__(self, model: str):
        self.model = model

    def match(self, request: Request) -> bool:
        return request.model == self.model
