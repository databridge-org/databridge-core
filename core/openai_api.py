from fastapi import APIRouter, HTTPException, Depends
import logging
import time
import uuid
from typing import List

from core.shared import verify_token, document_service, settings  # Import from shared module
from core.models.auth import AuthContext
from core.models.openai_models import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    ChatMessage,
    ChatCompletionChoice,
    CompletionChoice,
    EmbeddingData,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/openai", tags=["OpenAI"])


def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format a list of chat messages into a single prompt string."""
    formatted_messages = []
    for msg in messages:
        role_prefix = {"system": "System:", "user": "Human:", "assistant": "Assistant:"}.get(
            msg.role, f"{msg.role}:"
        )
        formatted_messages.append(f"{role_prefix} {msg.content}")
    return "\n".join(formatted_messages)


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count based on words."""
    return len(text.split())


@router.post("/chat/completions", response_model=CreateChatCompletionResponse)
async def create_chat_completion(
    request: CreateChatCompletionRequest, auth: AuthContext = Depends(verify_token)
) -> CreateChatCompletionResponse:
    """Create a chat completion following the OpenAI API format."""
    try:
        # Format messages into a prompt
        prompt = format_chat_messages(request.messages)

        # Call our document service query
        completion_response = await document_service.query(
            query=prompt,
            auth=auth,
            filters={},
            k=4,  # number of context chunks to retrieve
            min_score=0.0,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_reranking=True,
        )

        # Create OpenAI-style response
        response = CreateChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion_response.completion),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": estimate_tokens(prompt),
                "completion_tokens": estimate_tokens(completion_response.completion),
                "total_tokens": estimate_tokens(prompt)
                + estimate_tokens(completion_response.completion),
            },
        )
        return response
    except Exception as e:
        logger.exception("Error in chat completion endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions", response_model=CreateCompletionResponse)
async def create_completion(
    request: CreateCompletionRequest, auth: AuthContext = Depends(verify_token)
) -> CreateCompletionResponse:
    """Create a text completion following the OpenAI API format."""
    try:
        # Handle both string and list prompts
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

        # Call our document service query
        completion_response = await document_service.query(
            query=prompt,
            auth=auth,
            filters={},
            k=2,  # fewer chunks for simple completion
            min_score=0.0,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_reranking=True,
        )

        # Create OpenAI-style response
        response = CreateCompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    text=completion_response.completion,
                    index=0,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": estimate_tokens(prompt),
                "completion_tokens": estimate_tokens(completion_response.completion),
                "total_tokens": estimate_tokens(prompt)
                + estimate_tokens(completion_response.completion),
            },
        )
        return response
    except Exception as e:
        logger.exception("Error in text completion endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings", response_model=CreateEmbeddingResponse)
async def create_embedding(
    request: CreateEmbeddingRequest, auth: AuthContext = Depends(verify_token)
) -> CreateEmbeddingResponse:
    """Create embeddings following the OpenAI API format."""
    try:
        # Handle both string and list inputs
        inputs = [request.input] if isinstance(request.input, str) else request.input

        embeddings = []
        total_tokens = 0

        # Generate embeddings for each input
        for idx, text in enumerate(inputs):
            embedding = await document_service.embedding_model.embed_for_query(text)
            embeddings.append(EmbeddingData(embedding=embedding, index=idx))
            total_tokens += estimate_tokens(text)

        # Create OpenAI-style response
        response = CreateEmbeddingResponse(
            data=embeddings,
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )
        return response
    except Exception as e:
        logger.exception("Error in embeddings endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available models."""
    return {
        "data": [
            {
                "id": settings.COMPLETION_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "databridge",
            },
            {
                "id": settings.EMBEDDING_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "databridge",
            },
        ]
    }
