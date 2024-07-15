import time

from openai import OpenAI

from mocogpt import gpt_server


class TestChatCompletions:
    def test_should_reply_content_for_specified_prompt(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"
            assert response.usage.prompt_tokens >= 0
            assert response.usage.completion_tokens >= 0
            assert response.usage.total_tokens >= 0

    def test_should_reply_content_for_specified_messages(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(messages=[{"role": "user", "content": "Hi"}]).response(
            content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"
            assert response.usage.prompt_tokens >= 0
            assert response.usage.completion_tokens >= 0
            assert response.usage.total_tokens >= 0

    def test_should_reply_content_for_specified_prompt_in_stream(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(content="How can I assist you?")

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

    def test_should_reply_content_for_specified_prompt_and_model(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi", model="gpt-4").response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_temperature(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(temperature=1.0).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_max_token(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(max_tokens=4096).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=1.0,
                max_tokens=4096
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_user(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(user='u123456').response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                user='u123456'
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_stop(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(stop="STOP").response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                stop="STOP"
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_n(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(n=2).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                n=2
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_seed(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(seed=12345).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                seed=12345
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_frequency_penalty(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(frequency_penalty=1.0).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                frequency_penalty=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_presence_penalty(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(presence_penalty=1.0).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                presence_penalty=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_logprobs(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(logprobs=True).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                logprobs=True
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_top_logprobs(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(top_logprobs=3).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                top_logprobs=3
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_top_p(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(top_p=1.0).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                top_p=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_logit_bias(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(logit_bias={"1787": -100}).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                logit_bias={"1787": -100}
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_tools(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                tools=[
                    {
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["location"],
                            },
                        },
                        "type": "function"
                    }
                ]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_tool_choice(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(tool_choice="auto").response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                tool_choice="auto"
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_parallel_tool_calls(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(parallel_tool_calls=True).response(content="How can I assist you?")

        with server:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi"}],
                parallel_tool_calls=True
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_reply_content_for_specified_api_key(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(api_key="sk-123456789", prompt="Hi").response(content="Hi")
        server.chat.completions.request(api_key="sk-987654321", prompt="Hi").response(content="Hello")

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

    def test_should_reply_content_for_specified_organization_and_project(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(organization="123456", project="123456", prompt="Hi").response(content="Hi")
        server.chat.completions.request(organization="654321", project="654321", prompt="Hi").response(content="Hello")

        with server:
            client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                            organization="123456", project="123456")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "Hi"

            client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-987654321",
                            organization="654321", project="654321")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "Hello"

    def test_should_reply_content_for_specified_delay(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(content="How can I assist you?", sleep=1)

        with server:
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )
            stop = time.time()

            assert response.choices[0].message.content == "How can I assist you?"
            assert stop - start > 1

    def test_should_reply_content_for_finish_reason(self, client: OpenAI):
        server = gpt_server(12306)
        server.chat.completions.request(prompt="Hi").response(content="How can I assist you?", finish_reason="length")

        with server:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"
            assert response.usage.prompt_tokens >= 0
            assert response.usage.completion_tokens >= 0
            assert response.usage.total_tokens >= 0
            assert response.choices[0].finish_reason == "length"

