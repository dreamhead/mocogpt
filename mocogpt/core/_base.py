import random
import string
import time
from abc import ABC

import tiktoken


def generate_unique_id(length=30):
    timestamp = str(int(time.time() * 1000))
    random_part = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length - len(timestamp)))
    unique_id = timestamp + random_part
    return unique_id


def split_content(model, content):
    encoding = tiktoken.encoding_for_model(model)
    result = encoding.encode(content)
    return [encoding.decode_single_token_bytes(token).decode("utf-8") for token in result]


class Request:
    def __init__(self, headers, content: dict):
        self._headers = headers
        self._content = content

    @property
    def prompt(self) -> str:
        return self._content['messages'][-1]['content']

    @property
    def model(self) -> str:
        return self._content['model']

    @property
    def temperature(self) -> float:
        return self._content['temperature']

    @property
    def stream(self) -> bool:
        if 'stream' in self._content:
            return self._content['stream']

        return False

    @property
    def api_key(self) -> str:
        return self._headers['Authorization'].split('Bearer ')[1]


class Response:
    def __init__(self, model):
        self._id = f"chatcmpl-{generate_unique_id()}"
        self._model = model
        self._content = []

    def to_dict(self):
        return {
            'id': self._id,
            'created': int(time.time()),
            'model': self._model,
            'choices': self._choices()
        }

    def _choices(self):
        choices = []
        for index, content in enumerate(self._content):
            choices.append({
                "index": index,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "logprobs": None,
                "finish_reason": "stop"
            })
        return choices

    @property
    def model(self):
        return self._model

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, content):
        self._content.append(content)

    async def sse_content(self):
        created = int(time.time())
        for content in self._content:
            parts = split_content(self._model, content)
            for data in parts:
                yield {
                    "id": self._id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self._model,
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": data
                            },
                            "logprobs": None,
                            "finish_reason": "stop" if data == parts[-1] and content == self._content[-1] else None
                        }
                    ]
                }


class SessionContext:
    def __init__(self, request: Request):
        self._request = request
        self._response = Response(request.model)

    @property
    def response(self):
        return self._response


class RequestMatcher(ABC):
    def match(self, request: Request) -> bool:
        pass


class ResponseHandler(ABC):
    def write_response(self, context: SessionContext):
        pass


class SessionSetting:
    def __init__(self, matcher: RequestMatcher):
        self._matcher = matcher
        self._handler = None

    def response(self, handler):
        self._handler = handler
        return self


class GptServer(ABC):
    def completions(self, matcher) -> SessionSetting:
        pass

    def request(self, matcher) -> SessionSetting:
        pass
