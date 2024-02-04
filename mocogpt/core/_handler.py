from mocogpt.core._base import ResponseHandler, SessionContext


class ContentResponseHandler(ResponseHandler):
    def __init__(self, content: str):
        self.content = content

    def write_response(self, context: SessionContext):
        context.response.content = self.content


class EmbeddingsResponseHandler(ResponseHandler):
    def __init__(self, embedding: list[float]):
        self._embedding = embedding

    def write_response(self, context: SessionContext):
        context.response.embedding = self._embedding
