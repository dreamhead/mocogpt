import os
import subprocess
import time
from contextlib import contextmanager

import httpx
import openai
import pytest
from openai import OpenAI

from mocogpt.cli.app import __file__ as app_file


class TestMocoGPTCli:
    @contextmanager
    def run_service(self, filename, port):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, filename)
        service_process = subprocess.Popen(
            ["python", app_file, "start", config_file, "--port", port],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        try:
            yield service_process
        finally:
            service_process.terminate()
            service_process.wait()

    def test_should_run_with_prompt(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )
            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_run_with_model_and_prompt(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_run_with_temperature_and_prompt(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=1.0
            )

            assert response.choices[0].message.content == "How can I assist you?"

    def test_should_run_with_sleep_response(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "sleep"}]
            )
            stop = time.time()

            assert response.choices[0].message.content == "How can I assist you?"
            assert stop - start > 1

    def test_should_run_with_error_response(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            with pytest.raises(openai.RateLimitError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "error"}]
                )

    def test_should_run_with_redirect_response(self, client):
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                        http_client=httpx.Client(base_url="http://localhost:12306/v1", follow_redirects=False))

        with self.run_service("chat_completions_config.json", "12306"):
            with pytest.raises(openai.APIStatusError):
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "redirect"}]
                )

    def test_should_run_with_n(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-4",
                n=2,
                messages=[{"role": "user", "content": "Hi,n"}]
            )
            assert response.choices[0].message.content == "Hi, n"

    def test_should_run_with_seed(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi,seed"}],
                seed=12345
            )

            assert response.choices[0].message.content == "Hi, seed"

    def test_should_run_with_stop(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi,stop"}],
                stop="STOP"
            )

            assert response.choices[0].message.content == "Hi, stop"

    def test_should_run_with_frequency_penalty(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi, frequency_penalty"}],
                frequency_penalty=1.0
            )

            assert response.choices[0].message.content == "Hi, frequency_penalty"

    def test_should_run_with_presence_penalty(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi, presence_penalty"}],
                presence_penalty=1.0
            )

            assert response.choices[0].message.content == "Hi, presence_penalty"

    def test_should_run_with_top_p(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "user", "content": "Hi, top_p"}],
                top_p=1.0
            )

            assert response.choices[0].message.content == "Hi, top_p"

    def test_should_run_with_organization_and_project(self, client):
        with self.run_service("chat_completions_config.json", "12306"):
            client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789", organization="123456",
                            project="123456")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Organization and Project"}]
            )
            assert response.choices[0].message.content == "How can I assist you?"
