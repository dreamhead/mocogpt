import pytest
from openai import BadRequestError, OpenAI

from mocogpt import all_of, any_of, content, gpt_server, model, prompt, temperature


class TestMocoGPT:
    def test_should_reply_content_for_specified_prompt(self, client: OpenAI):
        server = gpt_server(12306)
        server.request(prompt("Hi")).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_prompt_in_stream(self, client: OpenAI):
        server = gpt_server(12306)
        server.request(prompt("Hi")).response(content("How can I assist you?"))

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
        server.request(all_of(prompt("Hi"), model("gpt-4"))).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_prompt_or_model(self, client: OpenAI):
        server = gpt_server(12306)
        server.request(any_of(prompt("Hi"), model("gpt-4"))).response(content("How can I assist you?"))

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
        server.request(temperature(1.0)).response(content("How can I assist you?"))

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

