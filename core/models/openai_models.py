from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
import time


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(
        ..., description="The role of the message author. Can be 'system', 'user', or 'assistant'."
    )
    content: str = Field(..., description="The content of the message.")
    name: Optional[str] = Field(None, description="The name of the author of this message.")
    function_call: Optional[Dict[str, Any]] = Field(
        None, description="The name and arguments of a function that should be called"
    )


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""

    index: int = Field(..., description="The index of this choice among all choices.")
    message: ChatMessage = Field(..., description="The chat completion message.")
    finish_reason: Optional[str] = Field(
        None, description="The reason the chat completion finished."
    )


class CreateChatCompletionRequest(BaseModel):
    """Request to create a chat completion."""

    model: str = Field(..., description="ID of the model to use.")
    messages: List[ChatMessage] = Field(
        ..., description="A list of messages comprising the conversation so far."
    )
    temperature: Optional[float] = Field(1.0, description="Sampling temperature between 0 and 2.")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter.")
    n: Optional[int] = Field(1, description="Number of chat completion choices to generate.")
    stream: Optional[bool] = Field(False, description="Whether to stream responses.")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Up to 4 sequences where the API will stop generating."
    )
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate.")
    presence_penalty: Optional[float] = Field(
        0.0, description="Penalty for new tokens based on their presence in text so far."
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Penalty for new tokens based on their frequency in text so far."
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify likelihood of specified tokens appearing."
    )
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user.")


class CreateChatCompletionResponse(BaseModel):
    """Response from creating a chat completion."""

    id: str = Field(..., description="A unique identifier for the chat completion.")
    object: Literal["chat.completion"] = Field(
        "chat.completion", description="The object type (always 'chat.completion')."
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp of when the chat completion was created.",
    )
    model: str = Field(..., description="The model used for completion.")
    choices: List[ChatCompletionChoice] = Field(..., description="The list of completion choices.")
    usage: Dict[str, int] = Field(..., description="Usage statistics for the completion request.")


class CompletionChoice(BaseModel):
    """A single text completion choice."""

    text: str = Field(..., description="The completed text.")
    index: int = Field(..., description="The index of this choice among all choices.")
    logprobs: Optional[Any] = Field(None, description="Log probabilities of tokens.")
    finish_reason: Optional[str] = Field(None, description="The reason the completion finished.")


class CreateCompletionRequest(BaseModel):
    """Request to create a text completion."""

    model: str = Field(..., description="ID of the model to use.")
    prompt: Union[str, List[str]] = Field(
        ..., description="The prompt(s) to generate completions for."
    )
    max_tokens: Optional[int] = Field(16, description="Maximum number of tokens to generate.")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature between 0 and 2.")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter.")
    n: Optional[int] = Field(1, description="Number of completions to generate.")
    stream: Optional[bool] = Field(False, description="Whether to stream responses.")
    logprobs: Optional[int] = Field(None, description="Include log probabilities of tokens.")
    echo: Optional[bool] = Field(
        False, description="Echo back the prompt in addition to the completion."
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Up to 4 sequences where the API will stop generating."
    )
    presence_penalty: Optional[float] = Field(
        0.0, description="Penalty for new tokens based on their presence in text so far."
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Penalty for new tokens based on their frequency in text so far."
    )
    best_of: Optional[int] = Field(
        1, description="Generate best_of completions server-side and return the best."
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify likelihood of specified tokens appearing."
    )
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user.")


class CreateCompletionResponse(BaseModel):
    """Response from creating a text completion."""

    id: str = Field(..., description="A unique identifier for the completion.")
    object: Literal["text_completion"] = Field(
        "text_completion", description="The object type (always 'text_completion')."
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp of when the completion was created.",
    )
    model: str = Field(..., description="The model used for completion.")
    choices: List[CompletionChoice] = Field(..., description="The list of completion choices.")
    usage: Dict[str, int] = Field(..., description="Usage statistics for the completion request.")


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: Literal["embedding"] = Field(
        "embedding", description="The object type (always 'embedding')."
    )
    embedding: List[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="The index of this embedding in the list.")


class CreateEmbeddingRequest(BaseModel):
    """Request to create embeddings."""

    model: str = Field(..., description="ID of the model to use.")
    input: Union[str, List[str]] = Field(..., description="Input text to get embeddings for.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user.")


class CreateEmbeddingResponse(BaseModel):
    """Response from creating embeddings."""

    object: Literal["list"] = Field("list", description="The object type (always 'list').")
    data: List[EmbeddingData] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="The model used for embedding.")
    usage: Dict[str, int] = Field(..., description="Usage statistics for the embedding request.")
