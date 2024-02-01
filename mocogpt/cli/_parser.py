from mocogpt import all_of, content, model, prompt, temperature

from ._server import console_server


class ConfigParser:
    def parse(self, cliargs):
        server = console_server(cliargs.port)
        for setting in cliargs.settings:
            self.bing_to(setting, server)

        return server

    def create_matcher(self, request):
        matchers = []
        if "prompt" in request:
            matchers.append(prompt(request['prompt']))

        if "model" in request:
            _model = request['model']
            if self.is_accepted_model(_model):
                matchers.append(model(_model))
            else:
                raise ValueError(f"Unsupported model: {_model}")

        if "temperature" in request:
            matchers.append(temperature(request['temperature']))

        if len(matchers) == 1:
            return matchers[0]

        return all_of(*matchers)

    def create_handler(self, response):
        handlers = []
        if "content" in response:
            handlers.append(content(response['content']))

        if len(handlers) == 1:
            return handlers[0]

    def bing_to(self, setting, server):
        matcher = None
        handler = None

        if "chat.completions" in setting:
            matcher = self.create_matcher(setting['chat.completions'])

        if 'response' in setting:
            handler = self.create_handler(setting['response'])

        if matcher is not None and handler is not None:
            server.chat.completions.on(matcher).response(handler)

    def is_accepted_model(self, model_name) -> bool:
        return model_name in [
            "gpt-4-0125-preview",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613"
        ]
