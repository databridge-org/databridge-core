import pytest
from openai import OpenAI
from typing import Generator
from fastapi.testclient import TestClient

from core.api import app
from core.shared import settings

# Test constants
TEST_API_KEY = "test-key-123"
TEST_PROMPT = "What is the capital of France?"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]


@pytest.fixture
def test_client() -> Generator:
    with TestClient(app) as client:
        yield client


@pytest.fixture
def openai_client() -> OpenAI:
    # Create OpenAI client pointed at our test server
    return OpenAI(
        api_key=TEST_API_KEY, base_url="http://localhost:8000/openai"  # This works with TestClient
    )


def test_chat_completion(test_client: TestClient, openai_client: OpenAI):
    """Test chat completion endpoint using OpenAI SDK."""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",  # This will be overridden by Databridge's model
        messages=[{"role": "user", "content": TEST_PROMPT}],
        temperature=0.7,
        max_tokens=150,
    )

    assert response.choices[0].message.content is not None
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens > 0


def test_completion(test_client: TestClient, openai_client: OpenAI):
    """Test text completion endpoint using OpenAI SDK."""
    response = openai_client.completions.create(
        model="text-davinci-003",  # This will be overridden by Databridge's model
        prompt=TEST_PROMPT,
        temperature=0.7,
        max_tokens=150,
    )

    assert response.choices[0].text is not None
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens > 0


def test_embeddings(test_client: TestClient, openai_client: OpenAI):
    """Test embeddings endpoint using OpenAI SDK."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",  # This will be overridden by Databridge's model
        input=TEST_PROMPT,
    )

    assert len(response.data) == 1
    assert len(response.data[0].embedding) > 0
    assert response.usage.total_tokens > 0


def test_list_models(test_client: TestClient, openai_client: OpenAI):
    """Test models endpoint using OpenAI SDK."""
    response = openai_client.models.list()

    # Should return at least our completion and embedding models
    assert len(response.data) >= 2
    model_ids = [model.id for model in response.data]
    assert settings.COMPLETION_MODEL in model_ids
    assert settings.EMBEDDING_MODEL in model_ids
