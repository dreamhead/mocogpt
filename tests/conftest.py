import pytest
from openai import OpenAI


@pytest.fixture
def client():
    client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789")
    yield client