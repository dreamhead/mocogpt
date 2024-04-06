import openai
import pytest
from openai import OpenAI

from mocogpt import (
    authentication_error,
    bad_request,
    conflict_error,
    gpt_server,
    internal_error,
    not_found,
    permission_denied,
    rate_limit,
    unprocessable_entity,
)


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

    def test_should_raise_bad_request_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(error=bad_request("Bad Request", 'new_api_error'))

        with server:
            with pytest.raises(openai.BadRequestError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_unprocessable_entity(self, client: OpenAI):
        server = gpt_server(12306)
        (server.chat.completions.request(prompt="Hi")
         .response(error=unprocessable_entity("Unprocessable Entity", 'new_api_error')))

        with server:
            with pytest.raises(openai.UnprocessableEntityError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_conflict_error(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(error=conflict_error("Conflict Error", 'new_api_error'))

        with server:
            with pytest.raises(openai.ConflictError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_internal_error(self, client: OpenAI):
        server = gpt_server(12306)
        (server.chat.completions.request(prompt="Hi")
         .response(error=internal_error("Internal Server Error", 'new_api_error')))

        with server:
            with pytest.raises(openai.InternalServerError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_raise_rate_limit_error_in_embeddings(self, client: OpenAI):
        server = gpt_server(12306)
        server.embeddings.request(input="Hi").response(error=rate_limit("Rate limit exceeded", 'new_api_error'))

        with server:
            with pytest.raises(openai.RateLimitError):
                client.embeddings.create(
                    model="text-embedding-ada-002",
                    input="Hi",
                    encoding_format="float"
                )

    def test_should_raise_authentication_error_in_embeddings(self, client: OpenAI):
        server = gpt_server(12306)
        server.embeddings.request(input="Hi").response(
            error=authentication_error("Authentication Error", 'new_api_error'))

        with server:
            with pytest.raises(openai.AuthenticationError):
                client.embeddings.create(
                    model="text-embedding-ada-002",
                    input="Hi",
                    encoding_format="float"
                )

