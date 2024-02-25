from ._args import StartArgs
from ._server import console_server


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
        if "content" in response:
            return {"content": response['content']}

        return {}


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
        if "embeddings" in response:
            return {"embeddings": response['embeddings']}

        return {}


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
