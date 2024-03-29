from .core.actual_server import ActualGptServer
from .core.base_server import GptServer
from .core.base_typing import any_of, contains, endswith, eq, none_of, regex, startswith, rate_limit, \
    authentication_error, permission_denied, not_found, bad_request

__all__ = [
    'eq',
    'any_of',
    'none_of',
    'contains',
    'startswith',
    'endswith',
    'regex',
    'gpt_server',
    'rate_limit',
    'authentication_error',
    'permission_denied',
    'not_found',
    'bad_request'
]


def gpt_server(port) -> GptServer:
    return ActualGptServer(port)
