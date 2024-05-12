import httpx
import openai
import pytest
from openai import OpenAI

from mocogpt import (
    gpt_server,
    redirect,
)


class TestMocoRedirect:
    def test_should_redirect_307(self, client: OpenAI):
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                        http_client=httpx.Client(base_url="http://localhost:12306/v1", follow_redirects=False))

        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(
            redirect=redirect(307, "http://localhost:12306/v1/chat/completions"))

        with server:
            with pytest.raises(openai.APIStatusError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_follow_redirect_307(self, client: OpenAI):
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                        http_client=httpx.Client(base_url="http://localhost:12306/v1", follow_redirects=True))

        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(
            redirect=redirect(307, "http://localhost:12307/v1/chat/completions"))

        next_server = gpt_server(12307)
        next_server.chat.completions.request(prompt="Hi").response(content="How can I assist you?")

        with server:
            with next_server:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )
                assert response.choices[0].message.content == "How can I assist you?"

    def test_should_redirect_307_in_embeddings(self, client: OpenAI):
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                        http_client=httpx.Client(base_url="http://localhost:12306/v1", follow_redirects=False))

        server = gpt_server(12306)
        server.embeddings.request(input="Hi").response(redirect=redirect(307, "http://localhost:12307/v1/chat/completions"))

        with server:
            with pytest.raises(openai.APIStatusError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}])
