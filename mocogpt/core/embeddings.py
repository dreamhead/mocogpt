from mocogpt.core.base_typing import Endpoint, Request, Response, ResponseHandler, SessionContext, SleepResponseHandler, \
    APIErrorHandler


class EmbeddingsRequest(Request):
    _content_fields = ['input', 'encoding_format', 'dimensions', 'user']


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


class EmbeddingsResponseHandler(ResponseHandler[EmbeddingsResponse]):
    def __init__(self, embedding: list[float]):
        self._embedding = embedding

    def write_response(self, context: SessionContext):
        context.response.embedding = self._embedding


class Embeddings(Endpoint):
    _request_params = {
        'api_key',
        'model',
        'input',
        'encoding_format',
        'dimensions',
        'user'
    }
    _response_params = {
        'embeddings': EmbeddingsResponseHandler,
        'sleep': SleepResponseHandler,
        'error': APIErrorHandler
    }
