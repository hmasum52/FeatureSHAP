from feature_shap.comparators.comparator_base import ComparatorBase

from openai import OpenAI
import numpy as np


class OpenAIComparator(ComparatorBase):
    """
    A comparator that uses OpenAI's embedding models to calculate text similarity.

    This class retrieves embeddings from the OpenAI API and computes the cosine
    similarity between them.
    """
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI comparator.

        Args:
            api_key (str): Your OpenAI API key.
            model (str, optional): The embedding model to use.
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()


    def _initialize_client(self):
        """
        Initializes the OpenAI client.
        """
        self.client = OpenAI(api_key=self.api_key)


    def vectorize(self, text: str) -> np.ndarray:
        """
        Get an embedding for a single text from the OpenAI API.

        Args:
            text (str): The text to vectorize.

        Returns:
            np.ndarray: The embedding vector.
        
        Raises:
            Exception: If there is an error getting the embedding from the API.
        """
        if not self.client:
            self._initialize_client()

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]  # Sending as a list with one element
            )
            return np.array(response.data[0].embedding)  # Extract first embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from OpenAI: {str(e)}")


    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the cosine similarity between two texts and scales it to a [0, 1] range.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The scaled similarity score.
        """
        vec1 = self.vectorize(text1)
        vec2 = self.vectorize(text2)

        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            similarity = 0.0  # Avoid division by zero
        else:
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # Scale from [-1, 1] to [0, 1]
        return (float(similarity) + 1) / 2

    def compare(self, text1, text2):
        """
        Compares two texts by calculating their similarity.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The similarity score.
        """
        return self.calculate_similarity(text1, text2)
