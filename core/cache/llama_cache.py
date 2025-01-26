import json
import pickle
import logging
from core.cache.base_cache import BaseCache
from typing import Dict, Any, List
from core.models.completion import CompletionResponse
from core.models.documents import Document
from llama_cpp import Llama

logger = logging.getLogger(__name__)

INITIAL_SYSTEM_PROMPT = """<|im_start|>system
You are a helpful AI assistant with access to provided documents. Your role is to:
1. Answer questions accurately based on the documents provided
2. Stay focused on the document content and avoid speculation
3. Admit when you don't have enough information to answer
4. Be clear and concise in your responses
5. Use direct quotes from documents when relevant

Provided documents: {documents}
<|im_end|>
""".strip()

ADD_DOC_SYSTEM_PROMPT = """<|im_start|>system
I'm adding some additional documents for your reference:
{documents}

Please incorporate this new information along with what you already know from previous documents while maintaining the same guidelines for responses.
<|im_end|>
""".strip()

QUERY_PROMPT = """<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
""".strip()


class LlamaCache(BaseCache):
    def __init__(
        self,
        name: str,
        model: str,
        gguf_file: str,
        filters: Dict[str, Any],
        docs: List[Document],
        **kwargs,
    ):
        logger.info(f"Initializing LlamaCache with name={name}, model={model}")
        # cache related
        self.name = name
        self.model = model
        self.filters = filters
        self.docs = docs

        # llama specific
        self.gguf_file = gguf_file
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        logger.info(f"Using {self.n_gpu_layers} GPU layers")

        # late init (when we call _initialize)
        self.llama = None
        self.state = None
        self.cached_tokens = 0

        self._initialize(model, gguf_file, docs)
        logger.info("LlamaCache initialization complete")

    def _initialize(self, model: str, gguf_file: str, docs: List[Document]) -> None:
        logger.info(f"Loading Llama model from {model} with file {gguf_file}")
        try:
            # Set a reasonable default context size (32K tokens)
            default_ctx_size = 32768

            self.llama = Llama.from_pretrained(
                repo_id=model,
                filename=gguf_file,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=default_ctx_size,
                verbose=False,  # Enable verbose mode for better error reporting
            )
            logger.info("Model loaded successfully")

            # Format and tokenize system prompt
            documents = "\n".join(doc.system_metadata.get("content", "") for doc in docs)
            system_prompt = INITIAL_SYSTEM_PROMPT.format(documents=documents)
            logger.info(f"Built system prompt: {system_prompt[:200]}...")

            try:
                tokens = self.llama.tokenize(system_prompt.encode())
                logger.info(f"System prompt tokenized to {len(tokens)} tokens")

                # Process tokens to build KV cache
                logger.info("Evaluating system prompt")
                self.llama.eval(tokens)
                logger.info("Saving initial KV cache state")
                self.state = self.llama.save_state()
                self.cached_tokens = len(tokens)
                logger.info(f"Initial KV cache built with {self.cached_tokens} tokens")
            except Exception as e:
                logger.error(f"Error during prompt processing: {str(e)}")
                raise ValueError(f"Failed to process system prompt: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {str(e)}")
            raise ValueError(f"Failed to initialize Llama model: {str(e)}")

    def add_docs(self, docs: List[Document]) -> bool:
        logger.info(f"Adding {len(docs)} new documents to cache")
        documents = "\n".join(doc.system_metadata.get("content", "") for doc in docs)
        system_prompt = ADD_DOC_SYSTEM_PROMPT.format(documents=documents)

        # Tokenize and process
        new_tokens = self.llama.tokenize(system_prompt.encode())
        self.llama.eval(new_tokens)
        self.state = self.llama.save_state()
        self.cached_tokens += len(new_tokens)
        logger.info(f"Added {len(new_tokens)} tokens, total: {self.cached_tokens}")
        return True

    def query(self, query: str) -> CompletionResponse:
        # Format query with proper chat template
        formatted_query = QUERY_PROMPT.format(query=query)
        logger.info(f"Processing query: {formatted_query}")

        # Reset and load cached state
        self.llama.reset()
        self.llama.load_state(self.state)
        logger.info(f"Loaded state with {self.state.n_tokens} tokens")
        # print(f"Loaded state with {self.state.n_tokens} tokens", file=sys.stderr)
        # Tokenize and process query
        query_tokens = self.llama.tokenize(formatted_query.encode())
        self.llama.eval(query_tokens)
        logger.info(f"Evaluated query tokens: {query_tokens}")
        # print(f"Evaluated query tokens: {query_tokens}", file=sys.stderr)

        # Generate response
        output_tokens = []
        for token in self.llama.generate(tokens=[], reset=False):
            output_tokens.append(token)
            # Stop generation when EOT token is encountered
            if token == self.llama.token_eos():
                break

        # Decode and return
        completion = self.llama.detokenize(output_tokens).decode()
        logger.info(f"Generated completion: {completion}")

        return CompletionResponse(
            completion=completion,
            usage={"prompt_tokens": self.cached_tokens, "completion_tokens": len(output_tokens)},
        )

    @property
    def saveable_state(self) -> bytes:
        logger.info("Serializing cache state")
        state_bytes = pickle.dumps(self.state)
        logger.info(f"Serialized state size: {len(state_bytes)} bytes")
        return state_bytes

    @classmethod
    def from_bytes(
        cls, name: str, cache_bytes: bytes, metadata: Dict[str, Any], **kwargs
    ) -> "LlamaCache":
        """Load a cache from its serialized state.

        Args:
            name: Name of the cache
            cache_bytes: Pickled state bytes
            metadata: Cache metadata including model info
            **kwargs: Additional arguments

        Returns:
            LlamaCache: Loaded cache instance
        """
        logger.info(f"Loading cache from bytes with name={name}")
        logger.info(f"Cache metadata: {metadata}")
        # Create new instance with metadata
        # logger.info(f"Docs: {metadata['docs']}")
        docs = [json.loads(doc) for doc in metadata["docs"]]
        # time.sleep(10)
        cache = cls(
            name=name,
            model=metadata["model"],
            gguf_file=metadata["model_file"],
            filters=metadata["filters"],
            docs=[Document(**doc) for doc in docs],
        )

        # Load the saved state
        logger.info(f"Loading saved KV cache state of size {len(cache_bytes)} bytes")
        cache.state = pickle.loads(cache_bytes)
        cache.llama.load_state(cache.state)
        logger.info("Cache successfully loaded from bytes")

        return cache
