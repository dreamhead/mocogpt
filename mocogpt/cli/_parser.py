import mocogpt

from ._args import StartArgs
from ._server import console_server


def create_error(error):
    name = error['name']
    type = error['type']
    message = error['message']
    api_error = getattr(mocogpt, name)
    if api_error and callable(api_error):
        return api_error(message, type)

    raise f"Unknown API Error: {name}"


def create_direct(redirect):
    status = redirect['status']
    location = redirect['location']
    return mocogpt.redirect(status, location)


def create_common_matcher(request: dict, matcher: dict):
    if "api_key" in request:
        matcher['api_key'] = request['api_key']

    if "organization" in request:
        matcher['organization'] = request['organization']

    if "project" in request:
        matcher['project'] = request['project']


def create_common_handler(response: dict, handler: dict):
    if "sleep" in response:
        handler["sleep"] = response['sleep']
    if "error" in response:
        handler["error"] = create_error(response["error"])
    if "redirect" in response:
        handler["redirect"] = create_direct(response["redirect"])


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

        create_common_matcher(request, matcher)

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

        create_common_handler(response, handler)

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

        create_common_matcher(request, matcher)

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

        create_common_handler(response, handler)

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
