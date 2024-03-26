import openai
import pytest
from openai import OpenAI

from mocogpt import gpt_server, rate_limit, authentication_error, permission_denied, not_found, bad_request, \
    conflict_error


class TestMocoGPT:
    def test_should_raise_rate_limit_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(error=rate_limit("Rate limit exceeded", 'new_api_error'))

        with server:
            with pytest.raises(openai.RateLimitError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_authentication_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(
            error=authentication_error("Authentication Error", 'new_api_error'))

        with server:
            with pytest.raises(openai.AuthenticationError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_permission_denied_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(
            error=permission_denied("Permission Denied", 'new_api_error'))

        with server:
            with pytest.raises(openai.PermissionDeniedError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_not_found_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(error=not_found("Not Found", 'new_api_error'))

        with server:
            with pytest.raises(openai.NotFoundError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_bad_request_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(error=bad_request("Bad Request", 'new_api_error'))

        with server:
            with pytest.raises(openai.BadRequestError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_conflict_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(error=conflict_error("Bad Request", 'new_api_error'))

        with server:
            with pytest.raises(openai.ConflictError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

