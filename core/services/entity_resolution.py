import httpx
import logging
import json
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, ConfigDict

from core.config import get_settings
from core.models.graph import Entity
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# Define Pydantic models for structured output
class EntityGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical: str
    variants: List[str]


class EntityResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_groups: List[EntityGroup]


class EntityResolver:
    """
    Resolves and normalizes entities by identifying different variants of the same entity.
    Handles cases like "Trump" vs "Donald J Trump" or "JFK" vs "Kennedy".
    """

    def __init__(self):
        """Initialize the entity resolver"""
        self.settings = get_settings()

    async def resolve_entities(self, entities: List[Entity]) -> Tuple[List[Entity], Dict[str, str]]:
        """
        Resolves entities by identifying and grouping entities that refer to the same real-world entity.

        Args:
            entities: List of extracted entities

        Returns:
            Tuple containing:
            - List of normalized entities
            - Dictionary mapping original entity text to canonical entity text
        """
        if not entities:
            return [], {}

        # Extract entity labels to deduplicate
        entity_labels = [e.label for e in entities]

        # If there's only one entity, no need to resolve
        if len(entity_labels) <= 1:
            return entities, {entity_labels[0]: entity_labels[0]} if entity_labels else {}
        
        # Use LLM to identify and group similar entities
        resolved_entities = await self._resolve_with_llm(entity_labels)
        
        # Create mapping from original to canonical forms
        entity_mapping = {}
        for group in resolved_entities:
            canonical = group["canonical"]
            for variant in group["variants"]:
                entity_mapping[variant] = canonical
        
        # Deduplicate entities based on mapping
        unique_entities = []
        label_to_entity_map = {}
        
        # First, create a map from labels to entities
        for entity in entities:
            canonical_label = entity_mapping.get(entity.label, entity.label)
            
            # If we haven't seen this canonical label yet, add the entity to our map
            if canonical_label not in label_to_entity_map:
                label_to_entity_map[canonical_label] = entity
            else:
                # If we have seen it, merge chunk sources
                existing_entity = label_to_entity_map[canonical_label]
                
                # Merge document IDs
                for doc_id in entity.document_ids:
                    if doc_id not in existing_entity.document_ids:
                        existing_entity.document_ids.append(doc_id)
                
                # Merge chunk sources
                for doc_id, chunk_numbers in entity.chunk_sources.items():
                    if doc_id not in existing_entity.chunk_sources:
                        existing_entity.chunk_sources[doc_id] = list(chunk_numbers)
                    else:
                        for chunk_num in chunk_numbers:
                            if chunk_num not in existing_entity.chunk_sources[doc_id]:
                                existing_entity.chunk_sources[doc_id].append(chunk_num)
                
                # Merge properties (optional)
                for key, value in entity.properties.items():
                    if key not in existing_entity.properties:
                        existing_entity.properties[key] = value
    
        # Now, update the labels and create unique entities list
        for canonical_label, entity in label_to_entity_map.items():
            # Update the entity label to the canonical form
            entity.label = canonical_label
            
            # Add aliases property if there are variants
            variants = [variant for variant, canon in entity_mapping.items() 
                        if canon == canonical_label and variant != canonical_label]
            if variants:
                if 'aliases' not in entity.properties:
                    entity.properties['aliases'] = []
                entity.properties['aliases'].extend(variants)
                # Deduplicate aliases
                entity.properties['aliases'] = list(set(entity.properties['aliases']))
            
            unique_entities.append(entity)
            
        return unique_entities, entity_mapping
    
    async def _resolve_with_llm(self, entity_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Uses LLM to identify and group similar entities.
        
        Args:
            entity_labels: List of entity labels to resolve
            
        Returns:
            List of entity groups, where each group is a dict with:
            - "canonical": The canonical form of the entity
            - "variants": List of variant forms of the entity
        """
        prompt = self._create_entity_resolution_prompt(entity_labels)
        
        if self.settings.GRAPH_PROVIDER == "openai":
            return await self._resolve_with_openai(prompt, entity_labels)
        elif self.settings.GRAPH_PROVIDER == "ollama":
            return await self._resolve_with_ollama(prompt, entity_labels)
        else:
            logger.error(f"Unsupported graph provider: {self.settings.GRAPH_PROVIDER}")
            # Fallback: treat each entity as unique
            return [{"canonical": label, "variants": [label]} for label in entity_labels]

    async def _resolve_with_openai(self, prompt: str, entity_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Resolves entities using OpenAI.

        Args:
            prompt: Prompt for entity resolution
            entity_labels: Original entity labels

        Returns:
            List of entity groups
        """
        # Define the schema directly instead of converting from Pydantic model
        json_schema = {
            "type": "object",
            "properties": {
                "entity_groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical": {"type": "string"},
                            "variants": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["canonical", "variants"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entity_groups"],
            "additionalProperties": False
        }
        
        # System and user messages
        system_message = {
            "role": "system",
            "content": "You are an entity resolution expert. Your task is to identify and group different representations of the same real-world entity."
        }
        
        user_message = {
            "role": "user",
            "content": prompt
        }
        
        try:
            client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
            
            response = await client.responses.create(
                model=self.settings.GRAPH_MODEL,
                input=[system_message, user_message],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "entity_resolution",
                        "schema": json_schema,
                        "strict": True
                    }
                }
            )
            
            # Process response based on current API structure
            if hasattr(response, 'output_text') and response.output_text:
                content = response.output_text
                
                try:
                    # Parse with Pydantic model for validation
                    parsed_data = json.loads(content)
                    
                    # Validate with Pydantic model
                    resolution_result = EntityResolutionResult.model_validate(parsed_data)
                    
                    # Extract entity groups
                    return [group.model_dump() for group in resolution_result.entity_groups]
                    
                except Exception as e:
                    logger.error("Error parsing entity resolution response: %r", e)
                    
                    # Fallback to direct JSON parsing if Pydantic validation fails
                    try:
                        parsed_data = json.loads(content)
                        return parsed_data.get("entity_groups", [])
                    except Exception:
                        logger.error("JSON parsing failed, falling back to default resolution")
                        
            elif hasattr(response, 'refusal') and response.refusal:
                logger.warning("OpenAI refused to resolve entities")
                        
        except Exception as e:
            logger.error("Error during entity resolution with OpenAI: %r", e)
        
        # Fallback: treat each entity as unique
        logger.info("Falling back to treating each entity as unique")
        return [{"canonical": label, "variants": [label]} for label in entity_labels]
    
    async def _resolve_with_ollama(self, prompt: str, entity_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Resolves entities using Ollama.
        
        Args:
            prompt: Prompt for entity resolution
            entity_labels: Original entity labels
            
        Returns:
            List of entity groups
        """
        system_message = {
            "role": "system",
            "content": "You are an entity resolution expert. Your task is to identify and group different representations of the same real-world entity."
        }
        
        user_message = {
            "role": "user",
            "content": prompt
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Create the schema for structured output
                format_schema = EntityResolutionResult.model_json_schema()
                
                response = await client.post(
                    f"{self.settings.EMBEDDING_OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": self.settings.GRAPH_MODEL,
                        "messages": [system_message, user_message],
                        "stream": False,
                        "format": format_schema
                    },
                )
                response.raise_for_status()
                result = response.json()
                
                # Parse with Pydantic model for validation
                resolution_result = EntityResolutionResult.model_validate_json(result["message"]["content"])
                return [group.model_dump() for group in resolution_result.entity_groups]
                
        except Exception as e:
            logger.error("Error during entity resolution with Ollama: %r", e)
            
        # Fallback: treat each entity as unique
        return [{"canonical": label, "variants": [label]} for label in entity_labels]
    
    def _create_entity_resolution_prompt(self, entity_labels: List[str]) -> str:
        """
        Creates a prompt for the LLM to resolve entities.
        
        Args:
            entity_labels: List of entity labels to resolve
            
        Returns:
            Prompt string for the LLM
        """
        entities_str = "\n".join([f"- {label}" for label in entity_labels])
        entities_example_dict = {
            "entity_groups": [
                {
                    "canonical": "John F. Kennedy",
                    "variants": ["John F. Kennedy", "JFK", "Kennedy"]
                },
                {
                    "canonical": "United States of America",
                    "variants": ["United States of America", "USA", "United States"]
                }
            ]
        }
        
        prompt = f"""
Below is a list of entities extracted from a document:

{entities_str}

Some of these entities may refer to the same real-world entity but with different names or spellings.
For example, "JFK" and "John F. Kennedy" refer to the same person.

Please analyze this list and group entities that refer to the same real-world entity.
For each group, provide:
1. A canonical (standard) form of the entity
2. All variant forms found in the list

Format your response as a JSON object with an "entity_groups" array, where each item in the array is an object with:
- "canonical": The canonical form (choose the most complete and formal name)
- "variants": Array of all variants (including the canonical form)

The exact format of the JSON structure should be:
```json
{str(entities_example_dict)}
```

Only include entities in your response that have multiple variants or are grouped with other entities.
If an entity has no variants and doesn't belong to any group, don't include it in your response.

Focus on identifying:
- Different names for the same person (e.g., full names vs. nicknames)
- Different forms of the same organization
- The same concept expressed differently
- Abbreviations and their full forms
- Spelling variations and typos
"""
        return prompt
