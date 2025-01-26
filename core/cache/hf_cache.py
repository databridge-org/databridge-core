# hugging face cache implementation.

from core.cache.base_cache import BaseCache
from typing import List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from core.models.completion import CompletionRequest, CompletionResponse


class HuggingFaceCache(BaseCache):
    """Hugging Face Cache implementation for cache-augmented generation"""

    def __init__(
        self,
        cache_path: Path,
        model_name: str = "distilgpt2",
        device: str = "cpu",
        default_max_new_tokens: int = 100,
        use_fp16: bool = False,
    ):
        """Initialize the HuggingFace cache.

        Args:
            cache_path: Path to store cache files
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on (e.g. "cpu", "cuda", "mps")
            default_max_new_tokens: Default maximum number of new tokens to generate
            use_fp16: Whether to use FP16 precision
        """
        super().__init__()
        self.cache_path = cache_path
        self.model_name = model_name
        self.device = device
        self.default_max_new_tokens = default_max_new_tokens
        self.use_fp16 = use_fp16

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure model loading based on device
        model_kwargs = {"low_cpu_mem_usage": True}

        if device == "cpu":
            # For CPU, use standard loading
            model_kwargs.update({"torch_dtype": torch.float32})
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
        else:
            # For GPU/MPS, use automatic device mapping and optional FP16
            model_kwargs.update(
                {"device_map": "auto", "torch_dtype": torch.float16 if use_fp16 else torch.float32}
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.kv_cache = None
        self.origin_len = None

    def get_kv_cache(self, prompt: str) -> DynamicCache:
        """Build KV cache from prompt"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        cache = DynamicCache()

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return cache

    def clean_up_cache(self, cache: DynamicCache, origin_len: int):
        """Clean up cache by removing appended tokens"""
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

    def generate(
        self, input_ids: torch.Tensor, past_key_values, max_new_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model and cache"""
        device = next(self.model.parameters()).device
        origin_len = input_ids.shape[-1]
        input_ids = input_ids.to(device)
        output_ids = input_ids.clone()
        next_token = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens or self.default_max_new_tokens):
                out = self.model(
                    input_ids=next_token, past_key_values=past_key_values, use_cache=True
                )
                logits = out.logits[:, -1, :]
                token = torch.argmax(logits, dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, token], dim=-1)
                past_key_values = out.past_key_values
                next_token = token.to(device)

                if (
                    self.model.config.eos_token_id is not None
                    and token.item() == self.model.config.eos_token_id
                ):
                    break

        return output_ids[:, origin_len:]

    async def ingest(self, docs: List[str]) -> bool:
        """Ingest documents into cache"""
        try:
            # Create system prompt with documents
            system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers.
<|user|>
Context:
{' '.join(docs)}
Question:
""".strip()

            # Build the cache
            input_ids = self.tokenizer(system_prompt, return_tensors="pt").input_ids.to(self.device)
            self.kv_cache = DynamicCache()

            with torch.no_grad():
                # First run to get the cache shape
                outputs = self.model(input_ids=input_ids, use_cache=True)
                # Initialize cache with empty tensors of the right shape
                n_layers = len(outputs.past_key_values)
                batch_size = input_ids.shape[0]

                # Handle different model architectures

                if hasattr(self.model.config, "num_key_value_heads"):
                    # Models with grouped query attention (GQA) like Llama
                    n_kv_heads = self.model.config.num_key_value_heads
                    head_dim = self.model.config.head_dim
                elif hasattr(self.model.config, "n_head"):
                    # GPT-style models
                    n_kv_heads = self.model.config.n_head
                    head_dim = self.model.config.n_embd // self.model.config.n_head
                elif hasattr(self.model.config, "num_attention_heads"):
                    # OPT-style models
                    n_kv_heads = self.model.config.num_attention_heads
                    head_dim = (
                        self.model.config.hidden_size // self.model.config.num_attention_heads
                    )
                else:
                    raise ValueError(
                        f"Unsupported model architecture: {self.model.config.model_type}"
                    )

                seq_len = input_ids.shape[1]

                for i in range(n_layers):
                    key_shape = (batch_size, n_kv_heads, seq_len, head_dim)
                    value_shape = key_shape
                    self.kv_cache.key_cache.append(torch.zeros(key_shape, device=self.device))
                    self.kv_cache.value_cache.append(torch.zeros(value_shape, device=self.device))

                # Now run with the initialized cache
                outputs = self.model(
                    input_ids=input_ids, past_key_values=self.kv_cache, use_cache=True
                )
                # Update cache with actual values
                self.kv_cache.key_cache = [layer[0] for layer in outputs.past_key_values]
                self.kv_cache.value_cache = [layer[1] for layer in outputs.past_key_values]
                self.origin_len = self.kv_cache.key_cache[0].shape[-2]
            return True
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            return False

    async def update(self, new_doc: str) -> bool:
        """Update cache with new document"""
        try:
            if self.kv_cache is None:
                return await self.ingest([new_doc])

            # Clean up existing cache
            self.clean_up_cache(self.kv_cache, self.origin_len)

            # Add new document to cache
            input_ids = self.tokenizer(new_doc + "\n", return_tensors="pt").input_ids.to(
                self.device
            )

            # First run to get the cache shape
            outputs = self.model(input_ids=input_ids, use_cache=True)
            # Initialize cache with empty tensors of the right shape
            n_layers = len(outputs.past_key_values)
            batch_size = input_ids.shape[0]

            # Handle different model architectures
            if hasattr(self.model.config, "num_key_value_heads"):
                # Models with grouped query attention (GQA) like Llama
                n_kv_heads = self.model.config.num_key_value_heads
                head_dim = self.model.config.head_dim
            elif hasattr(self.model.config, "n_head"):
                # GPT-style models
                n_kv_heads = self.model.config.n_head
                head_dim = self.model.config.n_embd // self.model.config.n_head
            elif hasattr(self.model.config, "num_attention_heads"):
                # OPT-style models
                n_kv_heads = self.model.config.num_attention_heads
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
            else:
                raise ValueError(f"Unsupported model architecture: {self.model.config.model_type}")

            seq_len = input_ids.shape[1]

            # Create a new cache for the update
            new_cache = DynamicCache()
            for i in range(n_layers):
                key_shape = (batch_size, n_kv_heads, seq_len, head_dim)
                value_shape = key_shape
                new_cache.key_cache.append(torch.zeros(key_shape, device=self.device))
                new_cache.value_cache.append(torch.zeros(value_shape, device=self.device))

            # Run with the initialized cache
            outputs = self.model(input_ids=input_ids, past_key_values=new_cache, use_cache=True)
            # Update cache with actual values
            self.kv_cache.key_cache = [layer[0] for layer in outputs.past_key_values]
            self.kv_cache.value_cache = [layer[1] for layer in outputs.past_key_values]
            return True
        except Exception as e:
            print(f"Error updating cache: {e}")
            return False

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using cache-augmented generation"""
        try:
            if self.kv_cache is None:
                raise ValueError("Cache not initialized. Please ingest documents first.")

            # Clean up cache
            self.clean_up_cache(self.kv_cache, self.origin_len)

            # Generate completion
            input_ids = self.tokenizer(request.query + "\n", return_tensors="pt").input_ids.to(
                self.device
            )
            gen_ids = self.generate(input_ids, self.kv_cache, max_new_tokens=request.max_tokens)
            completion = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            # Calculate token usage
            usage = {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(gen_ids[0]),
                "total_tokens": len(input_ids[0]) + len(gen_ids[0]),
            }

            return CompletionResponse(completion=completion, usage=usage)
        except Exception as e:
            print(f"Error generating completion: {e}")
            return CompletionResponse(
                completion=f"Error: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

    def save_cache(self) -> Path:
        """Save the KV cache to disk"""
        if self.kv_cache is None:
            raise ValueError("No cache to save")

        cache_dir = self.cache_path / "kv_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save key and value caches
        cache_data = {
            "key_cache": self.kv_cache.key_cache,
            "value_cache": self.kv_cache.value_cache,
            "origin_len": self.origin_len,
        }
        cache_path = cache_dir / "cache.pt"
        torch.save(cache_data, cache_path)
        return cache_path

    def load_cache(self, cache_path: Union[str, Path]) -> None:
        """Load KV cache from disk"""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found at {cache_path}")

        cache_data = torch.load(cache_path, map_location=self.device)

        self.kv_cache = DynamicCache()
        self.kv_cache.key_cache = cache_data["key_cache"]
        self.kv_cache.value_cache = cache_data["value_cache"]
        self.origin_len = cache_data["origin_len"]
