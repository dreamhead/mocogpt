import hashlib
import time

import tiktoken

from mocogpt.core.base_typing import (
    Endpoint,
    Request,
    Response,
    ResponseHandler,
    SessionContext,
)


def count_tokens(model: str, content: str) -> int:
    return len(tiktoken.encoding_for_model(model).encode(content))


class CompletionsRequest(Request):
    _content_fields = ['temperature', 'max_tokens', 'user', 'n', 'seed', 'stop',
                       'frequency_penalty', 'presence_penalty', 'logprobs', 'top_logprobs', 'top_p']

    @property
    def prompt(self) -> str:
        return self._content['messages'][-1]['content']

    def prompt_tokens(self):
        return sum(count_tokens(self.model, content['content']) for content in self._content['messages'])

    @property
    def stream(self) -> bool:
        if 'stream' in self._content:
            return self._content['stream']

        return False


def generate_unique_id():
    return hashlib.md5(str(time.time()).encode()).hexdigest()


def split_content(model, content):
    encoding = tiktoken.encoding_for_model(model)
    result = encoding.encode(content)
    return [encoding.decode_single_token_bytes(token).decode("utf-8") for token in result]


class CompletionsResponse(Response):
    def __init__(self, model, prompt_tokens):
        super().__init__(model)
        self._id = f"chatcmpl-{generate_unique_id()}"
        self.prompt_tokens = prompt_tokens

    def to_dict(self):
        completion_tokens = self.completion_tokens()

        return {
            'id': self._id,
            'created': int(time.time()),
            'model': self._model,
            'choices': self._choices(),
            'usage': {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": self.prompt_tokens + completion_tokens
            }
        }

    def completion_tokens(self) -> int:
        return sum(count_tokens(self.model, content) for content in self._content)

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


class ContentResponseHandler(ResponseHandler[CompletionsResponse]):
    def __init__(self, content: str):
        self.content = content

    def write_response(self, context: SessionContext):
        context.response.content = self.content


class Completions(Endpoint):
    _request_params = [
        'model',
        'prompt',
        'temperature',
        'max_tokens',
        'user',
        'stop',
        'n',
        'seed',
        'frequency_penalty',
        'presence_penalty',
        'logprobs',
        'top_logprobs',
        'top_p'
    ]

    _response_params = {
        'content': ContentResponseHandler
    }


class Chat:
    def __init__(self, completions: Completions):
        self._completions = completions

    @property
    def completions(self) -> Completions:
        return self._completions
