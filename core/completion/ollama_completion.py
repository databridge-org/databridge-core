from core.completion.base_completion import (
    BaseCompletionModel,
    CompletionRequest,
    CompletionResponse,
)
from ollama import AsyncClient


class OllamaCompletionModel(BaseCompletionModel):
    """Ollama completion model implementation"""

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.client = AsyncClient(host=base_url)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Ollama API"""
        # Construct prompt with context
        context = "\n\n".join(request.context_chunks)
        prompt = f"""You are a helpful assistant. Use the provided context to answer questions accurately.

Context:
{context}

Question: {request.query}"""

        # Call Ollama API
        response = await self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        )

        # Ollama doesn't provide token usage info, so we'll estimate based on characters
        completion_text = response["message"]["content"]
        char_to_token_ratio = 4  # Rough estimate
        estimated_prompt_tokens = len(prompt) // char_to_token_ratio
        estimated_completion_tokens = len(completion_text) // char_to_token_ratio

        return CompletionResponse(
            completion=completion_text,
            usage={
                "prompt_tokens": estimated_prompt_tokens,
                "completion_tokens": estimated_completion_tokens,
                "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
            },
        )