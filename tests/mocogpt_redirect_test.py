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



