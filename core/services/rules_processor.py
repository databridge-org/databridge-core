from typing import Dict, Any, List, Tuple
import logging
import json
from core.models.chunk import Chunk
from core.completion.ollama_completion import OllamaCompletionModel
from core.completion.openai_completion import OpenAICompletionModel

logger = logging.getLogger(__name__)


class RulesProcessor:
    """Service for applying rules to chunks after parsing but before embedding."""

    def __init__(self, completion_model: OllamaCompletionModel | OpenAICompletionModel):
        self.model = completion_model

    async def _apply_rules_to_text(self, text: str, rules: List[str]) -> Tuple[Dict[str, Any], str]:
        """Apply rules to text and return extracted metadata and modified text."""
        # Construct prompt for rule application
        prompt = f"""You are a helpful assistant that processes text according to rules. Your task is to:
1. Extract metadata based on the rules
2. Return the text with any modifications required by the rules
3. ALWAYS return your response in valid JSON format

Apply these rules to the given text:

Rules:
{chr(10).join(f'- {rule}' for rule in rules)}

Text:
{text}

Return a JSON object with:
1. 'metadata': Extracted information based on rules
2. 'modified_text': Text after applying redaction/modification rules

Example response format:
{{
    "metadata": {{"extracted_name": "John Doe", "address": "123 Main St"}},
    "modified_text": <text after applying rules>
}}

Remember: Your entire response must be valid JSON, and only JSON, no commentary or other text."""

        # Get completion from model
        if isinstance(self.model, OllamaCompletionModel):
            logger.info(f"Using Ollama model {self.model.model_name} for rules processing")
            response = await self.model.client.chat(
                model=self.model.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            completion = response["message"]["content"]
            logger.info(f"Got response from Ollama: {completion}")
        else:  # OpenAI
            logger.info(f"Using OpenAI model {self.model.model_name} for rules processing")
            response = await self.model.client.chat.completions.create(
                model=self.model.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            completion = response.choices[0].message.content
            logger.info(f"Got response from OpenAI: {completion}")

        try:
            # First try direct JSON parsing
            result = json.loads(completion)
            logger.info(f"Successfully parsed model response: {result}")
            return result["metadata"], result["modified_text"]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {e}")
            logger.warning(f"Raw completion was: {completion}")

            # Try to extract JSON using regex
            import re

            json_match = re.search(r"\{[\s\S]*\}", completion)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    logger.info(f"Successfully extracted and parsed JSON from response: {result}")
                    return result["metadata"], result["modified_text"]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")

            # If all parsing attempts fail, try a simpler prompt
            simple_prompt = f"""Extract metadata from this text according to these rules. Return ONLY a JSON object.

Rules:
{chr(10).join(f'- {rule}' for rule in rules)}

Text:
{text}

Return format:
{{
    "metadata": {{}},
    "modified_text": "original text"
}}"""

            if isinstance(self.model, OllamaCompletionModel):
                response = await self.model.client.chat(
                    model=self.model.model_name,
                    messages=[{"role": "user", "content": simple_prompt}],
                    stream=False,
                )
                completion = response["message"]["content"]
            else:  # OpenAI
                response = await self.model.client.chat.completions.create(
                    model=self.model.model_name,
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.0,
                )
                completion = response.choices[0].message.content

            try:
                result = json.loads(completion)
                logger.info(f"Successfully parsed response from simple prompt: {result}")
                return result["metadata"], result["modified_text"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"All parsing attempts failed. Final error: {e}")
                logger.error(f"Final raw completion was: {completion}")
                raise ValueError(
                    f"Failed to extract metadata from model response after multiple attempts: {e}"
                )

    async def process_chunks(self, chunks: List[Chunk], rules: List[str]) -> List[Chunk]:
        """Process chunks by applying rules to extract metadata and modify content."""
        if not rules:
            return chunks

        # Combine all chunk content
        full_text = "\n".join(chunk.content for chunk in chunks)

        # Apply rules
        try:
            metadata, modified_text = await self._apply_rules_to_text(full_text, rules)
        except Exception as e:
            logger.error(f"Error processing rules: {e}")
            return chunks

        # Create new chunks from modified text
        # Note: We maintain the same chunking strategy as the original chunks
        chunk_size = len(chunks[0].content) if chunks else 1000
        new_chunks = []
        for i in range(0, len(modified_text), chunk_size):
            chunk_text = modified_text[i : i + chunk_size]
            chunk = Chunk(content=chunk_text, metadata=metadata.copy())
            new_chunks.append(chunk)

        return new_chunks
