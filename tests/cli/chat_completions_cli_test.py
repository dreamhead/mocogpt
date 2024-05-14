import os
import subprocess
import time

import httpx
import openai
import pytest
from openai import OpenAI

from mocogpt.cli.app import __file__ as app_file


class TestMocoGPTCli:
    def test_should_run_with_prompt(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}]
            )
            assert response.choices[0].message.content == "How can I assist you?"
        finally:
            service_process.terminate()
            service_process.wait()

    def test_should_run_with_model_and_prompt(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.choices[0].message.content == "How can I assist you?"
        service_process.terminate()
        service_process.wait()

    def test_should_run_with_temperature_and_prompt(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=1.0
        )

        assert response.choices[0].message.content == "How can I assist you?"
        service_process.terminate()
        service_process.wait()

    def test_should_run_with_sleep_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "sleep"}]
        )
        stop = time.time()

        assert response.choices[0].message.content == "How can I assist you?"
        assert stop - start > 1
        service_process.terminate()
        service_process.wait()

    def test_should_run_with_error_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "error"}]
            )

        service_process.terminate()
        service_process.wait()

    def test_should_run_with_redirect_response(self, client):
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                        http_client=httpx.Client(base_url="http://localhost:12306/v1", follow_redirects=False))

        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.APIStatusError):
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "redirect"}]
            )

        service_process.terminate()
        service_process.wait()

