from .core.actual_server import ActualGptServer
from .core.base_server import GptServer
from .core.base_typing import (
    any_of,
    api_error,
    authentication_error,
    bad_request,
    conflict_error,
    contains,
    endswith,
    eq,
    internal_error,
    none_of,
    not_found,
    permission_denied,
    rate_limit,
    redirect,
    regex,
    startswith,
    unprocessable_entity,
)

__all__ = [
    'eq',
    'any_of',
    'none_of',
    'contains',
    'startswith',
    'endswith',
    'regex',
    'gpt_server',
    'api_error',
    'rate_limit',
    'authentication_error',
    'permission_denied',
    'not_found',
    'bad_request',
    'conflict_error',
    'internal_error',
    'unprocessable_entity',
    'redirect',
]


def gpt_server(port) -> GptServer:
    return ActualGptServer(port)
