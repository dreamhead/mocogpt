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


def create_matcher(request: dict, matcher: dict, keys: list):
    matcher.update({k: v for k, v in request.items() if k in keys})


def create_common_matcher(request: dict, matcher: dict):
    create_matcher(request, matcher, ['api_key', 'organization', 'project', 'model'])


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
        create_matcher(request, matcher, ["prompt", "temperature", "max_tokens", "user", "n"])

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
        create_matcher(request, matcher, ["input", "encoding_format", "dimensions", "user"])

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
