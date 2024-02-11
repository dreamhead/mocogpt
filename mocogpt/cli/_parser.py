from ._args import StartArgs
from ._server import console_server


class ConfigParser:
    def parse(self, cliargs: StartArgs):
        server = console_server(cliargs.port)
        self.bing_to(cliargs.settings, server)
        return server

    def create_matcher(self, request):
        api_key = None
        prompt = None
        model = None
        temperature = None
        if "api_key" in request:
            api_key = request['api_key']

        if "prompt" in request:
            prompt = request['prompt']

        if "model" in request:
            _model = request['model']
            if self.is_accepted_model(_model):
                model = _model
            else:
                raise ValueError(f"Unsupported model: {_model}")

        if "temperature" in request:
            temperature = request['temperature']

        return api_key, prompt, model, temperature

    def create_handler(self, response):
        if "content" in response:
            return response['content']

        return None

    def bing_to(self, setting, server):
        if "chat.completions" in setting:
            self.bing_chat_completion(setting['chat.completions'], server)

    def bing_chat_completion(self, settings, server):
        api_key, prompt, model, temperature = None, None, None, None
        content = None

        for setting in settings:
            if 'request' in setting:
                api_key, prompt, model, temperature = self.create_matcher(setting['request'])
            if 'response' in setting:
                content = self.create_handler(setting['response'])
            (server.chat.completions
             .request(api_key=api_key, prompt=prompt, model=model, temperature=temperature)
             .response(content=content))

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
