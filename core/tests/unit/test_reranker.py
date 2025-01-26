import pytest
from typing import List

from core.models.chunk import DocumentChunk
from core.reranker.flag_reranker import FlagReranker
from core.config import get_settings


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    return [
        DocumentChunk(
            document_id="1",
            content="The quick brown fox jumps over the lazy dog",
            embedding=[0.1] * 10,
            chunk_number=1,
            score=0.5,
        ),
        DocumentChunk(
            document_id="2",
            content="Python is a popular programming language",
            embedding=[0.2] * 10,
            chunk_number=1,
            score=0.7,
        ),
        DocumentChunk(
            document_id="3",
            content="Machine learning models help analyze data",
            embedding=[0.3] * 10,
            chunk_number=1,
            score=0.3,
        ),
    ]


@pytest.fixture
def reranker():
    """Fixture to create and reuse a flag reranker instance"""
    settings = get_settings()
    if not settings.USE_RERANKING:
        pytest.skip("Reranker is disabled in settings")
    return FlagReranker(
        model_name=settings.RERANKER_MODEL,
        device=settings.RERANKER_DEVICE,
        use_fp16=settings.RERANKER_USE_FP16,
        query_max_length=settings.RERANKER_QUERY_MAX_LENGTH,
        passage_max_length=settings.RERANKER_PASSAGE_MAX_LENGTH,
    )


@pytest.mark.asyncio
async def test_reranker_relevance(reranker, sample_chunks):
    """Test that reranker improves relevance for programming-related query"""
    if not get_settings().USE_RERANKING:
        pytest.skip("Reranker is disabled in settings")
    print("\n=== Testing Reranker Relevance ===")
    query = "What is Python programming language?"

    # Get reranked results
    reranked_chunks = await reranker.rerank(query, sample_chunks)
    print(f"\nQuery: {query}")
    for i, chunk in enumerate(reranked_chunks):
        print(f"{i+1}. Score: {chunk.score:.3f} - {chunk.content}")

    # The most relevant chunks should be about Python
    assert "Python" in reranked_chunks[0].content
    assert reranked_chunks[0].score > reranked_chunks[-1].score

    # Check that irrelevant content (fox/dog) is ranked lower
    fox_chunk_idx = next(i for i, c in enumerate(reranked_chunks) if "fox" in c.content.lower())
    assert fox_chunk_idx > 0  # Should not be first


@pytest.mark.asyncio
async def test_reranker_score_distribution(reranker, sample_chunks):
    """Test that reranker produces reasonable score distribution"""
    if not get_settings().USE_RERANKING:
        pytest.skip("Reranker is disabled in settings")
    print("\n=== Testing Score Distribution ===")
    query = "Tell me about machine learning and data science"

    # Get reranked results
    reranked_chunks = await reranker.rerank(query, sample_chunks)
    print(f"\nQuery: {query}")
    for i, chunk in enumerate(reranked_chunks):
        print(f"{i+1}. Score: {chunk.score:.3f} - {chunk.content}")

    # Check score properties
    scores = [c.score for c in reranked_chunks]
    assert all(0 <= s <= 1 for s in scores)  # Scores should be between 0 and 1
    assert len(set(scores)) > 1  # Should have different scores (not all same)

    # Verify ordering
    assert scores == sorted(scores, reverse=True)  # Should be in descending order

    # Most relevant chunk should be about ML/data science
    top_chunk = reranked_chunks[0]
    assert any(term in top_chunk.content.lower() for term in ["machine learning", "data science"])


@pytest.mark.asyncio
async def test_reranker_batch_scoring(reranker):
    """Test that reranker can handle multiple queries/passages efficiently"""
    if not get_settings().USE_RERANKING:
        pytest.skip("Reranker is disabled in settings")
    print("\n=== Testing Batch Scoring ===")
    texts = [
        "Python is a programming language",
        "Machine learning is a field of AI",
        "The quick brown fox jumps",
        "Data science uses statistical methods",
    ]
    queries = ["What is Python?", "Explain artificial intelligence", "Tell me about data analysis"]

    # Test multiple queries against multiple texts
    for query in queries:
        scores = await reranker.compute_score(query, texts)
        print(f"\nQuery: {query}")
        for text, score in zip(texts, scores):
            print(f"Score: {score:.3f} - {text}")
        assert len(scores) == len(texts)
        assert all(isinstance(s, float) for s in scores)
        assert all(0 <= s <= 1 for s in scores)


@pytest.mark.asyncio
async def test_reranker_empty_and_edge_cases(reranker, sample_chunks):
    """Test reranker behavior with empty or edge case inputs"""
    if not get_settings().USE_RERANKING:
        pytest.skip("Reranker is disabled in settings")
    print("\n=== Testing Edge Cases ===")

    # Empty chunks list
    result = await reranker.rerank("test query", [])
    assert result == []
    print("Empty chunks test passed")

    # Single chunk
    single_chunk = DocumentChunk(
        document_id="1",
        content="Test content",
        embedding=[0.1] * 768,
        chunk_number=1,
        score=0.5,
    )
    result = await reranker.rerank("test query", [single_chunk])
    assert len(result) == 1
    assert isinstance(result[0].score, float)
    print(f"Single chunk test passed - Score: {result[0].score:.3f}")

    # Empty query
    result = await reranker.rerank("", sample_chunks)
    assert len(result) == len(sample_chunks)
    print("Empty query test passed")


@pytest.mark.asyncio
async def test_reranker_consistency(reranker, sample_chunks):
    """Test that reranker produces consistent results for same input"""
    if not get_settings().USE_RERANKING:
        pytest.skip("Reranker is disabled in settings")
    print("\n=== Testing Consistency ===")
    query = "What is Python programming?"

    # Run reranking multiple times
    results1 = await reranker.rerank(query, sample_chunks)
    results2 = await reranker.rerank(query, sample_chunks)

    # Scores should be the same across runs
    scores1 = [c.score for c in results1]
    scores2 = [c.score for c in results2]
    print("\nScores from first run:", [f"{s:.3f}" for s in scores1])
    print("Scores from second run:", [f"{s:.3f}" for s in scores2])
    assert scores1 == scores2

    # Order should be preserved
    assert [c.document_id for c in results1] == [c.document_id for c in results2]
    print("Order consistency test passed")
