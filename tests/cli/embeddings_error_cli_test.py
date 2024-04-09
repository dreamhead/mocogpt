import os
import subprocess

import openai
import pytest

from mocogpt.cli.app import __file__ as app_file


class TestEmbeddingsCliError:
    def test_should_run_with_rate_limit_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "embeddings_error_cli.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.RateLimitError):
            client.embeddings.create(
                model="text-embedding-ada-002",
                input="rate limit",
                encoding_format="float"
            )

        service_process.terminate()
        service_process.wait()

    def test_should_run_with_authentication_error_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "embeddings_error_cli.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.AuthenticationError):
            client.embeddings.create(
                model="text-embedding-ada-002",
                input="authentication error",
                encoding_format="float"
            )

        service_process.terminate()
        service_process.wait()

    def test_should_run_with_permission_denied_response(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "embeddings_error_cli.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.PermissionDeniedError):
            client.embeddings.create(
                model="text-embedding-ada-002",
                input="permission denied",
                encoding_format="float"
            )

        service_process.terminate()
        service_process.wait()
