import openai
import pytest
from openai import OpenAI

from mocogpt import gpt_server, rate_limit


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

