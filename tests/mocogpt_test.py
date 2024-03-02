import pytest
from openai import OpenAI, BadRequestError

from mocogpt import gpt_server, any_of, none_of, contains, startswith, endswith


class TestMocoGPT:
    def test_should_raise_exception_for_unknown_config(self):
        server = gpt_server(12306)
        with pytest.raises(TypeError):
            server.chat.completions.request(unknown="Hi").response(content="How can I assist you?")

        with pytest.raises(TypeError):
            server.chat.completions.request(prompt="Hi").response(unknown="Hi")

    def test_should_not_reply_anything(self, client: OpenAI):
        server = gpt_server(12306)

        with server:
            with pytest.raises(BadRequestError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_reply_content_for_specified_prompt_or_model(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(
            prompt="Hello", model="gpt-4"
        ).or_request(
            prompt="Hi", model="gpt-3.5-turbo-1106"
        ).response(content="How can I assist you?")

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

    def test_should_reply_content_for_specified_any_prompt(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt=any_of("Hi", "Hello")).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_non_prompt(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt=none_of("Hi", "Hello")).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Nice to meet you"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

            with pytest.raises(BadRequestError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}]
                )

    def test_should_reply_content_for_contains_operator(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt=contains("Hi")).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi, how are you?"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_startswith_operator(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt=startswith("Hi")).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi, how are you?"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_endswith_operator(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt=endswith("Hi")).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "It's you. Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"


