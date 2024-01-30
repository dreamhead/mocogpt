from mocogpt import all_of, content, model, prompt, temperature
from mocogpt.cli._server import console_server


def is_accepted_model(model) -> bool:
    return model in [
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


def create_matcher(request):
    matchers = []
    if "prompt" in request:
        matchers.append(prompt(request['prompt']))

    if "model" in request:
        _model = request['model']
        if is_accepted_model(_model):
            matchers.append(model(_model))
        else:
            raise ValueError(f"Unsupported model: {_model}")

    if "temperature" in request:
        matchers.append(temperature(request['temperature']))

    if len(matchers) == 1:
        return matchers[0]

    return all_of(*matchers)


def create_handler(response):
    handlers = []
    if "content" in response:
        handlers.append(content(response['content']))

    if len(handlers) == 1:
        return handlers[0]


def bing_to(setting, server):
    request = setting['request']
    response = setting['response']
    matcher = None
    handler = None

    if 'request' in setting:
        matcher = create_matcher(request)

    if 'response' in setting:
        handler = create_handler(response)

    if matcher is not None and handler is not None:
        server.request(matcher).response(handler)


def parse(args):
    server = console_server(args.port)
    for setting in args.settings:
        bing_to(setting, server)

    return server
