import os
from openai import OpenAI
from typing import List
from .logger import Logger

logger = Logger(__name__)


class EmbeddingCalculator:
    """
    Uses an OpenAI client to produce embeddings for text.
    """

    def __init__(
            self,
            model_name="text-embedding-ada-002",
            encoding_format=None,
            dimensions=None,
            user=None
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.log("error", "API_KEY_MISSING", {"error": "OPENAI_API_KEY not found in environment variables"})
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.user = user

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Returns a list of floats for the embedding of a single text string.
        """
        payload = {
            "model": self.model_name,
            "input": [text],
        }
        optional_params = {
            "encoding_format": self.encoding_format,
            "dimensions": self.dimensions,
            "user": self.user,
        }
        for key, value in optional_params.items():
            if value is not None:
                payload[key] = value
        try:
            response = self.client.embeddings.create(**payload)
            embedding = response.data[0].embedding
            # Convert any numpy values to Python float
            return [float(val) for val in embedding]
        except Exception as e:
            logger.log("error", "EMBEDDING_FAILED", {"error": str(e)})
            return [0.0] * 1536  # Default embedding dimension

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts. Returns a list of lists of floats.
        """
        if not texts:
            return []

        embeddings = []
        for text in texts:
            embedding = self.get_text_embedding(text)
            embeddings.append(embedding)

        return embeddings
