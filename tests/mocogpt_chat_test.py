import pytest
from openai import BadRequestError, OpenAI

from mocogpt import all_of, any_of, api_key, content, gpt_server, model, prompt, temperature


class TestMocoGPTChat:
    def test_should_reply_content_for_specified_prompt(self, client: OpenAI):
        server = gpt_server(12306)
        server.completions(prompt("Hi")).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_prompt_in_stream(self, client: OpenAI):
        server = gpt_server(12306)
        server.completions(prompt("Hi")).response(content("How can I assist you?"))

        with server:
            stream = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True
            )

            result = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    result = result + chunk.choices[0].delta.content

            assert result == "How can I assist you?"

    def test_should_not_reply_anything(self, client: OpenAI):
        server = gpt_server(12306)

        with server:
            with pytest.raises(BadRequestError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_reply_content_for_specified_prompt_and_model(self, client: OpenAI):
        server = gpt_server(12306)
        server.completions(all_of(prompt("Hi"), model("gpt-4"))).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_prompt_or_model(self, client: OpenAI):
        server = gpt_server(12306)
        server.completions(any_of(prompt("Hi"), model("gpt-4"))).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_temperature(self, client: OpenAI):
        server = gpt_server(12306)
        server.completions(temperature(1.0)).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_api_key(self, client: OpenAI):
        server = gpt_server(12306)
        server.completions(all_of(api_key("sk-123456789"), prompt("Hi"))).response(content("Hi"))
        server.completions(all_of(api_key("sk-987654321"), prompt("Hi"))).response(content("Hello"))

        with server:
            client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "Hi"

            client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-987654321")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "Hello"
