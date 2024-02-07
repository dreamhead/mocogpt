from typing import Literal

from mocogpt.core.base_matcher import ApiKeyMatcher, ModelMatcher
from mocogpt.core.base_typing import Endpoint, Request, RequestMatcher, Response, ResponseHandler, SessionContext


class EmbeddingsRequest(Request):
    _content_fields = ['input', 'encoding_format']


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


class EncodingFormatMatcher(RequestMatcher[EmbeddingsRequest]):
    def __init__(self, encoding_format: Literal["float", "base64"]):
        self._encoding_format = encoding_format

    def match(self, request: EmbeddingsRequest) -> bool:
        return request.encoding_format == self._encoding_format


class EmbeddingsResponseHandler(ResponseHandler[EmbeddingsResponse]):
    def __init__(self, embedding: list[float]):
        self._embedding = embedding

    def write_response(self, context: SessionContext):
        context.response.embedding = self._embedding


class Embeddings(Endpoint):
    _request_params = {
        'api_key': ApiKeyMatcher,
        'model': ModelMatcher,
        'input': InputMatcher,
        'encoding_format': EncodingFormatMatcher,
    }
    _response_params = {
        'embeddings': EmbeddingsResponseHandler
    }
