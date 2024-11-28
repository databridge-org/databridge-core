from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from .base_parser import BaseParser
from langchain_unstructured import UnstructuredLoader
import os
import tempfile
import base64
from dotenv import load_dotenv
import anthropic
import logging

logger = logging.getLogger(__name__)

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

class UnstructuredAPIParser(BaseParser):
    def __init__(
        self,
        api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        parallel_threads: int = 5,
        api_url: str = "https://api.unstructuredapp.io"
    ):
        load_dotenv()
        self.api_key = api_key
        self.api_url = api_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parallel_threads = parallel_threads
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Token tracking
        self.token_counts = {
            'input': 0,
            'output': 0,
            'cache_read': 0,
            'cache_creation': 0
        }
        self.token_lock = threading.Lock()

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        """Generate contextual description for a chunk using Claude."""
        response = self.anthropic_client.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        }
                    ]
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        
        # Update token counts thread-safely
        with self.token_lock:
            self.token_counts['input'] += response.usage.input_tokens
            self.token_counts['output'] += response.usage.output_tokens
            self.token_counts['cache_read'] += response.usage.cache_read_input_tokens
            self.token_counts['cache_creation'] += response.usage.cache_creation_input_tokens
        
        return response.content[0].text, response.usage

    def process_chunk(self, args: tuple) -> dict:
        """Process a single chunk with contextualization."""
        doc_content, chunk = args
        context, usage = self.situate_context(doc_content, chunk)
        return {
            'chunk': chunk,
            'context': context
        }

    def parse(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Parse content using Unstructured API and split into contextualized chunks."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(metadata)) as temp_file:
                if metadata.get("is_base64", False):
                    temp_file.write(base64.b64decode(content))
                else:
                    temp_file.write(content.encode('utf-8'))
                temp_file_path = temp_file.name

            try:
                # Load and partition document
                loader = UnstructuredLoader(
                    file_path=temp_file_path,
                    partition_via_api=True,
                    api_key=self.api_key,
                    chunking_strategy="by_title"
                )
                elements = loader.load()
                logger.info(f"Loaded {len(elements)} elements from document")

                # Get full document text
                full_text = "\n\n".join(element.page_content for element in elements)
                logger.info(f"Full document length: {len(full_text)} characters")

                # Prepare chunks for concurrent processing
                chunk_args = [(full_text, element.page_content) for element in elements]
                
                # Process chunks concurrently
                contextualized_chunks = []
                with ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
                    # Submit all tasks
                    future_to_chunk = {
                        executor.submit(self.process_chunk, args): args 
                        for args in chunk_args
                    }
                    
                    # Process completed tasks with progress bar
                    for future in tqdm(
                        as_completed(future_to_chunk), 
                        total=len(chunk_args),
                        desc="Processing chunks"
                    ):
                        try:
                            result = future.result()
                            contextualized_chunks.append(
                                f"{result['chunk']}\n\n{result['context']}"
                            )
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")

                # Log token usage statistics
                total_tokens = (
                    self.token_counts['input'] + 
                    self.token_counts['cache_read'] + 
                    self.token_counts['cache_creation']
                )
                cache_savings = (
                    (self.token_counts['cache_read'] / total_tokens * 100)
                    if total_tokens > 0 else 0
                )
                
                logger.info(f"Parsing complete. Statistics:")
                logger.info(f"- Total chunks processed: {len(contextualized_chunks)}")
                logger.info(f"- Input tokens: {self.token_counts['input']}")
                logger.info(f"- Output tokens: {self.token_counts['output']}")
                logger.info(f"- Cache creation tokens: {self.token_counts['cache_creation']}")
                logger.info(f"- Cache read tokens: {self.token_counts['cache_read']}")
                logger.info(f"- Cache savings: {cache_savings:.2f}%")
                logger.info("Note: Cache read tokens come at a 90% discount!")

                return contextualized_chunks

            finally:
                # Clean up temp file
                os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            raise

    def _get_file_extension(self, metadata: Dict[str, Any]) -> str:
        """Get appropriate file extension based on metadata."""
        if "filename" in metadata:
            _, ext = os.path.splitext(metadata["filename"])
            return ext if ext else ".txt"
        return ".txt"
