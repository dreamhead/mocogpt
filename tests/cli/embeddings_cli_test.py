import os
import subprocess

import httpx
import openai
import pytest
from openai import OpenAI

from mocogpt.cli.app import __file__ as app_file


class TestMocoEmbeddingsCli:
    def test_should_run_with_prompt(self, client):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "embeddings_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input="Hi",
                encoding_format="float",
                dimensions=1536,
                user="user123456"
            )

            assert response.data[0].embedding[0] == 0.002253932
        finally:
            service_process.terminate()
            service_process.wait()

    def test_should_run_with_redirect_response(self, client):
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789",
                        http_client=httpx.Client(base_url="http://localhost:12306/v1", follow_redirects=False))

        current_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_directory, "embeddings_config.json")
        service_process = subprocess.Popen(
            ["python", app_file,
             "start", config_file,
             "--port", "12306"
             ])

        with pytest.raises(openai.APIStatusError):
            client.embeddings.create(
                model="text-embedding-ada-002",
                input="Redirect",
                encoding_format="float",
                dimensions=1536,
                user="user123456"
            )

        service_process.terminate()
        service_process.wait()
