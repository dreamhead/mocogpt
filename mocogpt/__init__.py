from .core.actual_server import ActualGptServer
from .core.base_server import GptServer


def gpt_server(port) -> GptServer:
    return ActualGptServer(port)