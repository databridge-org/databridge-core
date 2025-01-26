import pytest
from core.models.documents import Document
from core.cache.llama_cache import LlamaCache
from core.models.completion import CompletionResponse

# TEST_MODEL = "QuantFactory/Llama3.2-3B-Enigma-GGUF"
TEST_MODEL = "QuantFactory/Dolphin3.0-Llama3.2-1B-GGUF"
# TEST_MODEL = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
TEST_GGUF_FILE = "*Q4_K_S.gguf"
# TEST_GGUF_FILE = "*Q4_K_M.gguf"


def get_test_document():
    """Load the example.txt file as a test document."""
    # test_file = Path(__file__).parent.parent / "assets" / "example.txt"
    # with open(test_file, "r") as f:
    #     content = f.read()

    content = """
                In darkest hours of coding's fierce domain,
                Where bugs lurk deep in shadows, hard to find,
                Each error message brings fresh waves of pain,
                As stack traces drive madness through the mind.

                Through endless loops of print statements we wade,
                Debug flags raised like torches in the night,
                While segfaults mock each careful plan we made,
                And race conditions laugh at our plight.

                O rubber duck, my silent debugging friend,
                Your plastic gaze holds wisdom yet untold,
                As line by line we trace paths without end,
                Seeking that elusive bug of gold.

                Yet hope remains while coffee still flows strong,
                Through debugging hell, we'll debug right from wrong.
            """.strip()

    return Document(
        external_id="alice_ch1",
        owner={"id": "test_user", "name": "Test User"},
        content_type="text/plain",
        system_metadata={
            "content": content,
            "title": "Alice in Wonderland - Chapter 1",
            "source": "test_document",
        },
    )


@pytest.fixture
def llama_cache():
    """Create a LlamaCache instance with the test document."""
    doc = get_test_document()
    cache = LlamaCache(
        name="test_cache", model=TEST_MODEL, gguf_file=TEST_GGUF_FILE, filters={}, docs=[doc]
    )
    return cache


def test_basic_rag_capabilities(llama_cache):
    """Test that the cache can answer basic questions about the document content."""
    # Test question about whether ingestion is actually happening
    response = llama_cache.query(
        "Summarize the content of the document. Please respond in a single sentence. Summary: "
    )
    assert isinstance(response, CompletionResponse)
    # assert "alice" in response.completion.lower()

    # # Test question about a specific detail
    # response = llama_cache.query(
    #     "What did Alice see the White Rabbit do with its watch? Please respond in a single sentence. Answer: "
    # )
    # assert isinstance(response, CompletionResponse)
    # # assert "waistcoat-pocket" in response.completion.lower() or "looked at it" in response.completion.lower()

    # # Test question about character description
    # response = llama_cache.query(
    #     "How did Alice's size change during the story? Please respond in a single sentence. Answer: "
    # )
    # assert isinstance(response, CompletionResponse)
    # # assert any(phrase in response.completion.lower() for phrase in ["grew larger", "grew smaller", "nine feet", "telescope"])

    # # Test question about plot elements
    # response = llama_cache.query(
    #     "What was written on the bottle Alice found? Please respond in a single sentence. Answer: "
    # )
    # assert isinstance(response, CompletionResponse)
    # # assert "drink me" in response.completion.lower()


# def test_cache_memory_persistence(llama_cache):
#     """Test that the cache maintains context across multiple queries."""

#     # First query to establish context
#     llama_cache.query(
#         "What was Alice doing before she saw the White Rabbit? Please respond in a single sentence. Answer: "
#     )

#     # Follow-up query that requires remembering previous context
#     response = llama_cache.query(
#         "What book was her sister reading? Please respond in a single sentence. Answer: "
#     )
#     assert isinstance(response, CompletionResponse)
#     # assert "no pictures" in response.completion.lower() or "conversations" in response.completion.lower()


def test_adding_new_documents(llama_cache):
    """Test that the cache can incorporate new documents into its knowledge."""

    # Create a new document with additional content
    new_doc = Document(
        external_id="alice_ch2",
        owner={"id": "test_user", "name": "Test User"},
        content_type="text/plain",
        system_metadata={
            "content": "Alice found herself in a pool of tears. She met a Mouse swimming in the pool.",
            "title": "Alice in Wonderland - Additional Content",
            "source": "test_document",
        },
    )

    # Add the new document
    success = llama_cache.add_docs([new_doc])
    assert success

    # Query about the new content
    response = llama_cache.query(
        "What did Alice find in the pool of tears? Please respond in a single sentence. Answer: "
    )
    assert isinstance(response, CompletionResponse)
    assert "mouse" in response.completion.lower()


def test_cache_state_persistence():
    """Test that the cache state can be saved and loaded."""

    # Create initial cache
    doc = get_test_document()
    original_cache = LlamaCache(
        name="test_cache", model=TEST_MODEL, gguf_file=TEST_GGUF_FILE, filters={}, docs=[doc]
    )

    # Get the state
    state_bytes = original_cache.saveable_state
    # Save state bytes to temporary file
    import tempfile
    import os
    import pickle

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "cache.pkl")

        # Save to file
        with open(cache_file, "wb") as f:
            pickle.dump(state_bytes, f)

        # Load from file
        with open(cache_file, "rb") as f:
            loaded_state_bytes = pickle.load(f)

        # # Verify state bytes match
        # assert state_bytes == loaded_state_bytes
        # state_bytes = loaded_state_bytes  # Use loaded bytes for rest of test

    # Create new cache from state
    loaded_cache = LlamaCache.from_bytes(
        name="test_cache",
        cache_bytes=loaded_state_bytes,
        metadata={
            "model": TEST_MODEL,
            "model_file": TEST_GGUF_FILE,
            "filters": {},
            "docs": [doc.model_dump_json()],
        },
    )

    # Verify the loaded cache works
    response = loaded_cache.query(
        "Summarize the content of the document. Please respond in a single sentence. Summary: "
    )
    assert isinstance(response, CompletionResponse)
    assert "coding" in response.completion.lower() or "debug" in response.completion.lower()
    # assert "bottle" in response.completion.lower() and "drink me" in response.completion.lower()
