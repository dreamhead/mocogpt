import os
import subprocess

import openai
import pytest

from mocogpt.cli.app import __file__ as app_file


class TestChatCompletionCliError:
    def test_should_run_with_rate_limit_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_error_cli.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.RateLimitError):
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "rate limit"}]
            )

        service_process.terminate()
        service_process.wait()

    def test_should_run_with_authentication_error_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_error_cli.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.AuthenticationError):
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "authentication error"}]
            )

        service_process.terminate()
        service_process.wait()

    def test_should_run_with_permission_denied_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "chat_completions_error_cli.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.PermissionDeniedError):
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "permission denied"}]
            )

        service_process.terminate()
        service_process.wait()
