from .base_embedding_model import BaseEmbeddingModel
import os
import voyageai
from typing import List, Union
from dotenv import load_dotenv

class VoyageAIEmbeddingModel(BaseEmbeddingModel):
    """
    VoyageAI embedding model
    """
    load_dotenv()
    def __init__(self):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    async def embed_for_ingestion(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            batches = [[text]]
        else:
            # Split text into batches of 128
            batches = [text[i:i + 128] for i in range(0, len(text), 128)]
        embeddings = []
        for batch in batches:
            # dimensions of voyage 3 are 1024
            embeddings.extend(self.client.embed(batch, model="voyage-3").embeddings)
        return embeddings
    
    async def embed_for_query(self, text: str) -> List[float]:
        return self.client.embed([text], model="voyage-3").embeddings[0]



