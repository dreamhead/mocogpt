from ._args import StartArgs
from ._server import console_server


class ConfigParser:
    def parse(self, cliargs: StartArgs):
        server = console_server(cliargs.port)
        self.bing_to(cliargs.settings, server)
        return server

    def create_completions_matcher(self, request):
        api_key = None
        prompt = None
        model = None
        temperature = None
        if "api_key" in request:
            api_key = request['api_key']

        if "prompt" in request:
            prompt = request['prompt']

        if "model" in request:
            model = request['model']

        if "temperature" in request:
            temperature = request['temperature']

        return api_key, model, prompt, temperature

    def create_completions_handler(self, response):
        if "content" in response:
            return response['content']

        return None

    def bing_to(self, setting, server):
        if "chat.completions" in setting:
            self.bing_chat_completion(setting['chat.completions'], server)
        if "embeddings" in setting:
            self.bing_embeddings(setting['embeddings'], server)

    def bing_chat_completion(self, settings, server):
        api_key, prompt, model, temperature = None, None, None, None
        content = None

        for setting in settings:
            if 'request' in setting:
                api_key, model, prompt, temperature = self.create_completions_matcher(setting['request'])
            if 'response' in setting:
                content = self.create_completions_handler(setting['response'])
            (server.chat.completions
             .request(api_key=api_key, prompt=prompt, model=model, temperature=temperature)
             .response(content=content))

    def bing_embeddings(self, settings, server):
        api_key = None
        model = None
        _input = None
        embeddings = None
        for setting in settings:
            if 'request' in setting:
                api_key, model, _input = self.create_embeddings_matcher(setting['request'])
            if 'response' in setting:
                embeddings = self.create_embeddings_handler(setting['response'])
            (server.embeddings
             .request(api_key=api_key, input=_input, model=model)
             .response(embeddings=embeddings))

    def create_embeddings_matcher(self, request):
        api_key = None
        model = None
        _input = None

        if "api_key" in request:
            api_key = request['api_key']

        if "model" in request:
            model = request['model']

        if "input" in request:
            _input = request['input']

        return api_key, model, _input

    def create_embeddings_handler(self, response):
        if "embeddings" in response:
            return response["embeddings"]

        return None
