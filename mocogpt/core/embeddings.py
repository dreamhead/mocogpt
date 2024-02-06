from mocogpt.core.base_matcher import ApiKeyMatcher, ModelMatcher
from mocogpt.core.base_typing import Endpoint, Request, RequestMatcher, Response, ResponseHandler, SessionContext


class EmbeddingsRequest(Request):
    def __init__(self, headers, content: dict):
        super().__init__(headers, content)

    @property
    def input(self) -> str:
        return self._content['input']


class EmbeddingsResponse(Response):
    def __init__(self, model: str):
        super().__init__(model)
        self._embedding = None

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, embedding):
        self._embedding = embedding

    def to_embeddings(self):
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": self.embedding,
                    "index": 0
                }
            ],
            "model": self._model
        }


class InputMatcher(RequestMatcher[EmbeddingsRequest]):
    def __init__(self, input: str):
        self._input = input

    def match(self, request: EmbeddingsRequest) -> bool:
        return request.input == self._input


class EmbeddingsResponseHandler(ResponseHandler[EmbeddingsResponse]):
    def __init__(self, embedding: list[float]):
        self._embedding = embedding

    def write_response(self, context: SessionContext):
        context.response.embedding = self._embedding


class Embeddings(Endpoint):
    _request_params = ['api_key', 'model', 'input']
    _response_params = ['embeddings']
    _matcher_classes = {
        'api_key': ApiKeyMatcher,
        'model': ModelMatcher,
        'input': InputMatcher
    }
    _handler_classes = {
        'embeddings': EmbeddingsResponseHandler
    }
