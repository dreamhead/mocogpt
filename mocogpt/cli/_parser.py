from mocogpt import (
    authentication_error,
    bad_request,
    conflict_error,
    internal_error,
    not_found,
    permission_denied,
    rate_limit,
)

from ._args import StartArgs
from ._server import console_server


def create_error(error):
    name = error['name']
    type = error['type']
    message = error['message']
    if name == 'rate_limit':
        return rate_limit(message, type)
    elif name == 'authentication_error':
        return authentication_error(message, type)
    elif name == 'permission_denied':
        return permission_denied(message, type)
    elif name == 'not_found':
        return not_found(message, type)
    elif name == 'bad_request':
        return bad_request(message, type)
    elif name == 'conflict_error':
        return conflict_error(message, type)
    elif name == 'internal_error':
        return internal_error(message, type)


class ChatCompletionsBinder:
    def bind(self, settings, server):
        for setting in settings:
            matcher = {}
            handler = {}
            if 'request' in setting:
                matcher = self.create_matcher(setting['request'])
            if 'response' in setting:
                handler = self.create_handler(setting['response'])

            (server.chat.completions
             .request(**matcher)
             .response(**handler))

    def create_matcher(self, request) -> dict:
        matcher = {}
        if "api_key" in request:
            matcher['api_key'] = request['api_key']

        if "prompt" in request:
            matcher['prompt'] = request['prompt']

        if "model" in request:
            matcher['model'] = request['model']

        if "temperature" in request:
            matcher['temperature'] = request['temperature']

        if "max_tokens" in request:
            matcher['max_tokens'] = request['max_tokens']

        if "user" in request:
            matcher['user'] = request['user']

        return matcher

    def create_handler(self, response) -> dict:
        handler = {}
        if "content" in response:
            handler["content"] = response['content']

        if "sleep" in response:
            handler["sleep"] = response['sleep']

        if "error" in response:
            print(response["error"])
            handler["error"] = create_error(response["error"])

        return handler


class EmbeddingsBinder:
    def bind(self, settings, server):
        matcher = {}
        handler = {}

        for setting in settings:
            if 'request' in setting:
                matcher = self.create_matcher(setting['request'])
            if 'response' in setting:
                handler = self.create_handler(setting['response'])
            (server.embeddings
             .request(**matcher)
             .response(**handler))

    def create_matcher(self, request):
        matcher = {}

        if "api_key" in request:
            matcher["api_key"] = request['api_key']

        if "model" in request:
            matcher["model"] = request['model']

        if "input" in request:
            matcher["input"] = request['input']

        if "encoding_format" in request:
            matcher["encoding_format"] = request['encoding_format']

        if "dimensions" in request:
            matcher["dimensions"] = request['dimensions']

        if "user" in request:
            matcher["user"] = request['user']

        return matcher

    def create_handler(self, response):
        handler = {}
        if "embeddings" in response:
            handler["embeddings"] = response["embeddings"]

        if "sleep" in response:
            handler["sleep"] = response['sleep']

        if "error" in response:
            handler["error"] = create_error(response["error"])

        return handler



class ConfigParser:
    def parse(self, cliargs: StartArgs):
        server = console_server(cliargs.port)
        self.bind(cliargs.settings, server)
        return server

    def bind(self, setting, server):
        if "chat.completions" in setting:
            ChatCompletionsBinder().bind(setting['chat.completions'], server)
        if "embeddings" in setting:
            EmbeddingsBinder().bind(setting['embeddings'], server)
