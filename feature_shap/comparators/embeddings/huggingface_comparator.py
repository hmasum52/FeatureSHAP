from feature_shap.comparators.comparator_base import ComparatorBase
import numpy as np


class HuggingFaceComparator(ComparatorBase):
    """
    A comparator that uses a Hugging Face sentence-transformer model to compute similarity.

    This class uses sentence embeddings to represent the texts and then calculates
    the cosine similarity between them.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the HuggingFace comparator.

        Args:
            model_name (str, optional): The name of the sentence-transformer model.
            device (str, optional): The device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()


    def _initialize_model(self):
        """
        Initializes the sentence-transformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. Please install with 'pip install sentence-transformers'")


    def vectorize(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a single text input.

        Args:
            text (str): The text to vectorize.

        Returns:
            np.ndarray: The embedding vector.
        """
        if not self.model:
            self._initialize_model()
        return self.model.encode(text, convert_to_numpy=True)


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

        # np.dot is equivalent to cosine similarity for normalized vectors
        similarity = np.dot(vec1, vec2)

        # Scale from [-1, 1] to [0, 1]
        return (similarity + 1) / 2


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
